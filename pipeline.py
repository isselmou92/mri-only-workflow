#!/usr/bin/env python
"""
Complete MRI-Only Workflow
==========================
End-to-end pipeline: MR NIfTI -> Inference -> DICOM Export (sCT, GT, MR)

This script consolidates:
- 3D UNet inference (MR -> sCT)
- DICOM export for all three modalities (same Patient/Study/FrameOfReference)

All three DICOM series share:
- PatientID / PatientName (derived from input filename)
- StudyInstanceUID
- FrameOfReferenceUID

Usage:
    python pipeline.py \
        --config configs/default.yaml \
        --ckpt checkpoints/best.pt \
        --mr data/validation_dataset/MR/MZEI00027_MR_T1.nii.gz \
        --gt data/validation_dataset/CT/MZEI00027_CT_T1.nii.gz \
        --ref data/input/dicom_volume/reference.dcm \
        --out output/workflow_results
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simple_sct.config import load_config, choose_device
from simple_sct.models.simple_unet3d import UNet3D

# Optional SimpleITK for alignment (imported on demand)
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False


# =============================================================================
# NIfTI I/O helpers
# =============================================================================

def load_nifti(path: Path, reorient_to_canonical: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[object]]:
    """
    Load a NIfTI file.
    
    Returns:
        arr: (D, H, W) float32 array
        affine: NIfTI affine matrix
        header: NIfTI header
    """
    img = nib.load(str(path))
    if reorient_to_canonical:
        img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype=np.float32)
    return np.asarray(arr, dtype=np.float32), img.affine, img.header


def save_nifti(path: Path, volume: np.ndarray, affine=None, header=None) -> None:
    """Save a NIfTI file, copying affine/header when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    ni = nib.Nifti1Image(volume.astype(np.float32), affine=affine, header=header)
    nib.save(ni, str(path))


# =============================================================================
# SimpleITK Rigid Alignment
# =============================================================================

def rigid_align(
    fixed_path: Path,
    moving_path: Path,
    output_path: Path,
    max_iterations: int = 500,
    verbose: bool = True,
) -> Tuple[np.ndarray, object]:
    """
    Rigidly align moving image to fixed image using SimpleITK.
    
    Uses Euler3DTransform (3 rotations + 3 translations) with Mattes MI metric.
    Multi-start strategy tries both MOMENTS and GEOMETRY initializations.
    
    Args:
        fixed_path: Path to fixed/reference NIfTI (target space)
        moving_path: Path to moving NIfTI (to be aligned)
        output_path: Path to save aligned NIfTI
        max_iterations: Maximum optimizer iterations
        verbose: Print registration progress
    
    Returns:
        (aligned_np, transform): Aligned numpy array and the transform object
    """
    if not SITK_AVAILABLE:
        raise ImportError("SimpleITK is required for alignment. Install with: pip install SimpleITK")
    
    # Load images
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    
    if verbose:
        print(f"    Fixed shape: {fixed.GetSize()}, spacing: {fixed.GetSpacing()}")
        print(f"    Moving shape: {moving.GetSize()}, spacing: {moving.GetSpacing()}")
    
    # Multi-start: try both MOMENTS and GEOMETRY initialization
    init_methods = [
        ("MOMENTS", sitk.CenteredTransformInitializerFilter.MOMENTS),
        ("GEOMETRY", sitk.CenteredTransformInitializerFilter.GEOMETRY),
    ]
    
    best_transform = None
    best_metric = float("inf")
    best_label = ""
    
    for label, init_type in init_methods:
        if verbose:
            print(f"    Trying {label} initialization...")
        
        try:
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving,
                sitk.Euler3DTransform(),
                init_type,
            )
        except RuntimeError as e:
            if verbose:
                print(f"      Initialization failed ({e}), skipping.")
            continue
        
        # Setup registration
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.25, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-6,
            numberOfIterations=max_iterations,
            gradientMagnitudeTolerance=1e-8,
            relaxationFactor=0.5,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        
        # Multi-resolution
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        reg.SetInitialTransform(initial_transform, inPlace=False)
        
        # Execute
        try:
            final_transform = reg.Execute(fixed, moving)
            final_metric = reg.GetMetricValue()
            
            if verbose:
                print(f"      Final MI: {final_metric:.6f}")
            
            if final_metric < best_metric:
                best_metric = final_metric
                best_transform = final_transform
                best_label = label
        except Exception as e:
            if verbose:
                print(f"      Registration failed: {e}")
            continue
    
    if best_transform is None:
        raise RuntimeError("All registration attempts failed.")
    
    if verbose:
        print(f"    Best: {best_label} (MI = {best_metric:.6f})")
    
    # Resample moving onto fixed grid
    resampled = sitk.Resample(
        moving, fixed, best_transform,
        sitk.sitkLinear, -1000.0, moving.GetPixelID()
    )
    
    # Save aligned NIfTI
    sitk.WriteImage(resampled, str(output_path))
    
    # Convert to numpy (Z,Y,X) -> (X,Y,Z)
    aligned_np = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0).astype(np.float32)
    
    # Extract Euler3D transform for reporting
    euler = best_transform
    try:
        if hasattr(best_transform, "GetNumberOfTransforms"):
            for i in range(best_transform.GetNumberOfTransforms()):
                t = best_transform.GetNthTransform(i)
                if "Euler3D" in t.GetName():
                    euler = sitk.Euler3DTransform(t)
                    break
        else:
            euler = sitk.Euler3DTransform(best_transform)
    except RuntimeError:
        pass
    
    if verbose and hasattr(euler, "GetParameters"):
        params = euler.GetParameters()
        if len(params) >= 6:
            print(f"    Rotation (rad): ({params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f})")
            print(f"    Translation: ({params[3]:.2f}, {params[4]:.2f}, {params[5]:.2f})")
    
    return aligned_np, euler


# =============================================================================
# Scaling helpers
# =============================================================================

def scale_input(x: np.ndarray, mode: str) -> np.ndarray:
    """Scale input MR volume for inference."""
    if mode == "zscore":
        m = float(x.mean())
        s = float(x.std()) + 1e-8
        return (x - m) / s
    if mode == "minmax":
        xmin = float(x.min())
        xmax = float(x.max()) + 1e-8
        u = (x - xmin) / (xmax - xmin)
        return u * 2.0 - 1.0
    if mode == "fg_percentile":
        from simple_sct.data.intensity import fg_percentile_normalize
        return fg_percentile_normalize(x)
    if mode == "none":
        return x.astype(np.float32)
    raise ValueError(f"Unknown scale_input: {mode}")


def invert_target_scaling(y_net: np.ndarray, mode: str, hu_min: float, hu_max: float) -> np.ndarray:
    """
    Convert network output back to interpretable units.
    - If training used 'hu', prediction is already HU (identity).
    - If training used 'minmax', map from [-1,1] back to [hu_min,hu_max].
    """
    if mode == "hu":
        return y_net.astype(np.float32)
    if mode == "minmax":
        u = 0.5 * (y_net + 1.0)
        return (u * (hu_max - hu_min) + hu_min).astype(np.float32)
    raise ValueError(f"Unknown scale_target: {mode}")


# =============================================================================
# Filename / text parsing helpers
# =============================================================================

def _clean_series_descr(text: str) -> str:
    """Mirror the splitter: take part after '/', remove spaces and hyphens."""
    t = str(text) if text is not None else ""
    if "/" in t:
        t = t.split("/", 1)[-1]
    return re.sub(r"[ \-]", "", t)


def _patient_part_from_name(pname: str) -> str:
    """Mirror the splitter: PatientName may contain '/', take the 3rd token if possible."""
    try:
        return str(pname).split("/")[2].strip()
    except Exception:
        return str(pname).strip()


def _series_part_from_desc(sdesc: str) -> str:
    try:
        part = str(sdesc).split("/")[1].strip()
    except Exception:
        part = str(sdesc).strip()
    return re.sub(r"[ \-]", "", part)


def _series_date_formatted(series_date: Optional[str]) -> str:
    sd = str(series_date) if series_date else ""
    return f"{sd[:4]}-{sd[4:6]}-{sd[6:]}" if len(sd) == 8 else ""


def _recon_tokens_from_filename(path_like: Path) -> Tuple[str, str]:
    """Extract ReconstructionAlgorithm / Number from the filename."""
    parts = os.path.splitext(os.path.basename(str(path_like)))[0].split("_")
    recon_alg = parts[2] if len(parts) > 2 else ""
    recon_num = parts[3] if len(parts) > 3 else ""
    return recon_alg, recon_num


def extract_mouse_id(filename: str) -> str:
    """
    Extract mouse ID from filename (e.g., 'MZEI00027_MR_T1.nii.gz' -> 'MZEI00027').
    """
    stem = Path(filename).stem
    if stem.endswith('.nii'):
        stem = stem[:-4]
    # Take first part before underscore
    parts = stem.split('_')
    return parts[0] if parts else stem


# =============================================================================
# Reference DICOM geometry extraction
# =============================================================================

def _get_ref_pixelspacing(ref) -> List[float]:
    try:
        if hasattr(ref, "SharedFunctionalGroupsSequence"):
            return list(map(float, ref.SharedFunctionalGroupsSequence[0][(0x0028, 0x9110)][0][(0x0028, 0x0030)].value))
        v = ref.get("PixelSpacing")
        return list(map(float, v)) if v is not None else [1.0, 1.0]
    except Exception:
        return [1.0, 1.0]


def _get_ref_slicethickness(ref) -> float:
    try:
        if hasattr(ref, "SharedFunctionalGroupsSequence"):
            return float(ref.SharedFunctionalGroupsSequence[0][(0x0028, 0x9110)][0][(0x0018, 0x0050)].value)
        v = ref.get("SliceThickness")
        return float(v) if v is not None else 1.0
    except Exception:
        return 1.0


def _get_ref_iop(ref) -> List[float]:
    try:
        if hasattr(ref, "SharedFunctionalGroupsSequence"):
            return list(map(float, ref.SharedFunctionalGroupsSequence[0][(0x0020, 0x9116)][0][(0x0020, 0x0037)].value))
        v = ref.get("ImageOrientationPatient")
        return list(map(float, v)) if v is not None else [1, 0, 0, 0, 1, 0]
    except Exception:
        return [1, 0, 0, 0, 1, 0]


def _get_ref_ipp_first(ref) -> List[float]:
    """Try Enhanced PerFrame[0] IPP, else classic ImagePositionPatient."""
    try:
        if hasattr(ref, "PerFrameFunctionalGroupsSequence"):
            return list(map(float, ref.PerFrameFunctionalGroupsSequence[0][(0x0020, 0x9113)][0][(0x0020, 0x0032)].value))
        v = ref.get("ImagePositionPatient")
        return list(map(float, v)) if v is not None else [0.0, 0.0, 0.0]
    except Exception:
        return [0.0, 0.0, 0.0]


# =============================================================================
# Volume preprocessing
# =============================================================================

def _rotate_180_posterior(vol_xyz: np.ndarray) -> np.ndarray:
    """
    Rotate volume by 180° around the left–right (X) axis (i.e., in posterior direction).
    This is equivalent to a 180° rotation in the Y–Z plane.
    In array terms (X,Y,Z): np.rot90(vol, k=2, axes=(1,2)).
    """
    return np.rot90(vol_xyz, k=2, axes=(1, 2))


# =============================================================================
# NiftiSeriesMeta - metadata container for DICOM export
# =============================================================================

class NiftiSeriesMeta:
    """
    Collects tags written into each per-slice DICOM; NO affine usage from NIfTI.
    Naming & orientation behaviors mirror the initial DICOM splitter.
    """

    DICOM_OBJECTS_UID = UID("1.2.826.0.1.3680043.1.2.100.6.40.0.76")

    def __init__(
        self,
        nifti_path: Path,
        reference_dicom: Path,
        patient_id: str,
        study_uid: str,
        frame_of_reference_uid: str,
        is_mr: bool = False,
    ):
        self.nifti_path = Path(nifti_path)
        self.is_mr = is_mr

        # Load NIfTI -> pixels only (no affine)
        nii = nib.load(str(self.nifti_path))
        data = nii.get_fdata()
        if data.ndim != 3:
            raise ValueError(f"NIfTI must be 3D, got shape {data.shape}")
        
        if is_mr:
            # For MR: Scale to reasonable integer range for display
            # Don't use RescaleSlope/Intercept as many viewers ignore it for MR
            data_min = float(data.min())
            data_max = float(data.max())
            data_range = data_max - data_min
            
            if data_range > 0:
                # Scale to 0-4000 range (typical MR display range)
                # Higher values = brighter (tissue), lower = darker (background)
                target_min = 0
                target_max = 4000
                scaled = (data - data_min) / data_range * (target_max - target_min) + target_min
                vol_xyz = np.round(scaled).astype(np.int16)
            else:
                vol_xyz = np.zeros(data.shape, dtype=np.int16)
            
            # Don't use RescaleSlope for MR - store raw display values
            self.mr_rescale_slope = 1.0
            self.mr_rescale_intercept = 0.0
        else:
            # For CT: clamp to int16 (assume HU-like values)
            vol_xyz = np.clip(np.round(data), -32768, 32767).astype(np.int16)
            self.mr_rescale_slope = 1.0
            self.mr_rescale_intercept = 0.0

        # Pre-rotate 180° in posterior direction (about X axis)
        self.vol_xyz = _rotate_180_posterior(vol_xyz)
        self.Z = self.vol_xyz.shape[2]  # number of slices

        # Reference DICOM template
        ref = pydicom.dcmread(str(reference_dicom))
        self.ref = ref

        # Geometry from reference
        self.PixelSpacing = _get_ref_pixelspacing(ref)
        self.SliceThickness = _get_ref_slicethickness(ref)
        self.IOP_original = _get_ref_iop(ref)
        self.IPP_first = _get_ref_ipp_first(ref)

        # Copy essential non-geometry tags (shared across series)
        self.StudyDate = getattr(ref, "StudyDate", "20000101")
        self.StudyTime = getattr(ref, "StudyTime", "000000")
        self.SeriesDate = getattr(ref, "SeriesDate", self.StudyDate)
        self.SeriesTime = getattr(ref, "SeriesTime", self.StudyTime)
        self.ContentDate = getattr(ref, "ContentDate", self.SeriesDate)
        self.ContentTime = getattr(ref, "ContentTime", self.SeriesTime)
        self.SeriesDateFormatted = _series_date_formatted(self.SeriesDate)

        self.Manufacturer = getattr(ref, "Manufacturer", "")
        self.ManufacturerModelName = getattr(ref, "ManufacturerModelName", "")
        self.ReferringPhysicianName = str(getattr(ref, "PatientName", "")).split("/")[0].split("^")[0].strip()

        self.ImageComments = getattr(ref, "ImageComments", "")

        # Tube / exposure (mirror splitter fallbacks)
        try:
            if hasattr(ref, "SharedFunctionalGroupsSequence"):
                fg = ref.SharedFunctionalGroupsSequence[0]
                self.KVP = fg[(0x0018, 0x9325)][0][(0x0018, 0x0060)].value
                self.ExposureTimeInms = fg[(0x0018, 0x9321)][0][(0x0018, 0x9328)].value
                self.XrayTubeCurrentMa = fg[(0x0018, 0x9321)][0][(0x0018, 0x9330)].value
                self.ExposureInmAs = fg[(0x0018, 0x9321)][0][(0x0018, 0x9332)].value
            else:
                self.KVP = getattr(ref, "KVP", 120)
                self.ExposureTimeInms = getattr(ref, "ExposureTime", 1000)
                self.XrayTubeCurrentMa = getattr(ref, "XRayTubeCurrent", 100)
                self.ExposureInmAs = getattr(ref, "Exposure", 100)
        except Exception:
            self.KVP, self.ExposureTimeInms, self.XrayTubeCurrentMa, self.ExposureInmAs = (120, 1000, 100, 100)
        self.XRayTubeCurrentInuA = float(self.XrayTubeCurrentMa) * 1000.0

        # Frame of Reference & Study/Patient (use provided shared UIDs)
        self.FrameOfReferenceUID = frame_of_reference_uid
        self.StudyInstanceUID = study_uid

        # StudyDescription GUI policy
        pat_part = _patient_part_from_name(getattr(ref, "PatientName", ""))
        sdesc_orig = getattr(ref, "StudyDescription", "")
        self.StudyDescription = f"{sdesc_orig}_{pat_part}_{self.SeriesDateFormatted}"
        _m = re.search(r"T[0-9].*", self.StudyDescription)
        self.StudyID = _m.group(0) if _m else None

        # Patient policy: PatientName == PatientID
        self.PatientID = str(patient_id)
        self.PatientName = self.PatientID
        self.PatientBirthDate = getattr(ref, "PatientBirthDate", "20000101")
        self.PatientSex = getattr(ref, "PatientSex", "O")

        # ProtocolName per splitter
        recon_alg, recon_num = _recon_tokens_from_filename(Path(reference_dicom))
        in_plane_um = int(round(float(self.PixelSpacing[0]) * 1000.0))
        series_descr_clean = _clean_series_descr(getattr(ref, "SeriesDescription", "CT"))
        self.ProtocolName = (
            f"{series_descr_clean}-{int(self.KVP)}kVp{int(round(self.XRayTubeCurrentInuA,0))}uA-"
            f"{in_plane_um}um-{recon_alg}"
        )

        # SeriesDescription per splitter
        ser_part_clean = _series_part_from_desc(getattr(ref, "SeriesDescription", "CT"))
        if recon_num:
            self.SeriesDescription = f"{pat_part}_{ser_part_clean}-recon{recon_num}"
        else:
            self.SeriesDescription = f"{pat_part}_{ser_part_clean}"

        # Bits/Photometric/Window & StationName policy
        self.BitsAllocated = 16
        self.BitsStored = 16
        self.HighBit = 15
        self.SamplesPerPixel = 1
        self.PixelRepresentation = 1
        self.PhotometricInterpretation = "MONOCHROME2"
        self.WindowCenter = getattr(ref, "WindowCenter", -300)
        self.WindowWidth = getattr(ref, "WindowWidth", 1300)
        self.StationName = self.ManufacturerModelName

        # file-meta (classic CT Image Storage)
        self.SOPClassUID = pydicom.uid.CTImageStorage
        self.file_meta = FileMetaDataset()
        self.file_meta.FileMetaInformationVersion = b"\x00\x01"
        self.file_meta.MediaStorageSOPClassUID = self.SOPClassUID
        self.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        self.file_meta.ImplementationClassUID = self.DICOM_OBJECTS_UID
        self.file_meta.ImplementationVersionName = "DicomObjects.NET"


# =============================================================================
# DICOM writing primitives
# =============================================================================

def _base_ds(meta: NiftiSeriesMeta, filename: str, modality: str = "CT") -> FileDataset:
    """Create base DICOM dataset with common tags."""
    # Create a copy of file_meta to avoid modifying the original
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = meta.file_meta.FileMetaInformationVersion
    file_meta.TransferSyntaxUID = meta.file_meta.TransferSyntaxUID
    file_meta.ImplementationClassUID = meta.file_meta.ImplementationClassUID
    file_meta.ImplementationVersionName = meta.file_meta.ImplementationVersionName
    
    # Set appropriate SOPClassUID based on modality
    if modality == "MR":
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        sop_class_uid = pydicom.uid.MRImageStorage
    else:
        file_meta.MediaStorageSOPClassUID = meta.file_meta.MediaStorageSOPClassUID
        sop_class_uid = meta.SOPClassUID
    
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = sop_class_uid
    
    ds.Modality = modality
    ds.StudyInstanceUID = meta.StudyInstanceUID
    ds.FrameOfReferenceUID = meta.FrameOfReferenceUID

    # Study / patient
    ds.StudyDescription = meta.StudyDescription
    ds.StudyID = meta.StudyID
    ds.PatientID = meta.PatientID
    ds.PatientName = meta.PatientName
    ds.PatientBirthDate = meta.PatientBirthDate
    ds.PatientSex = meta.PatientSex

    # Dates
    ds.StudyDate = meta.StudyDate
    ds.StudyTime = meta.StudyTime
    ds.SeriesDate = meta.SeriesDate
    ds.SeriesTime = meta.SeriesTime
    ds.ContentDate = meta.ContentDate
    ds.ContentTime = meta.ContentTime

    # Manufacturer / acquisition
    ds.Manufacturer = meta.Manufacturer
    ds.ManufacturerModelName = meta.ManufacturerModelName
    ds.ReferringPhysicianName = meta.ReferringPhysicianName
    ds.StationName = meta.StationName

    # Modality-specific tags
    if modality == "MR":
        # MR-specific tags (similar to Paravision output)
        ds.RescaleIntercept = meta.mr_rescale_intercept
        ds.RescaleSlope = meta.mr_rescale_slope
        # Don't set Window for MR - let viewer auto-window
        ds.ImageType = ["ORIGINAL", "PRIMARY"]
        # MR-specific acquisition parameters
        ds.MagneticFieldStrength = 9.4  # Tesla (typical preclinical)
        ds.RepetitionTime = 7200.0  # ms
        ds.EchoTime = 33.0  # ms
        ds.FlipAngle = 90.0
        ds.ImagingFrequency = 400.0  # MHz
    else:
        # CT-specific tags
        ds.KVP = meta.KVP
        ds.ExposureTimeInms = meta.ExposureTimeInms
        ds.ExposureInmAs = meta.ExposureInmAs
        ds.XRayTubeCurrentInuA = meta.XRayTubeCurrentInuA
        ds.WindowCenter = meta.WindowCenter
        ds.WindowWidth = meta.WindowWidth
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]

    ds.PhotometricInterpretation = meta.PhotometricInterpretation

    # Pixel semantics
    ds.BitsAllocated = meta.BitsAllocated
    ds.BitsStored = meta.BitsStored
    ds.HighBit = meta.HighBit
    ds.SamplesPerPixel = meta.SamplesPerPixel
    ds.PixelRepresentation = meta.PixelRepresentation

    # Protocol name per slice
    ds.ProtocolName = meta.ProtocolName
    return ds


def _write_instance(
    ds: FileDataset,
    pixels_2d: np.ndarray,
    series_uid: str,
    series_number: str,
    series_description: str,
    iop: List[float],
    ipp: List[float],
    pixel_spacing: List[float],
    slice_thickness: float,
    patient_position: str,
    instance_index_1based: int,
    save_path: Path,
):
    """Write a single DICOM slice instance."""
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.SeriesDescription = series_description

    ds.Rows, ds.Columns = pixels_2d.shape
    ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    ds.SliceThickness = float(slice_thickness)
    ds.NumberOfFrames = 1

    ds.PatientPosition = patient_position
    ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.ImagePositionPatient = [float(x) for x in ipp]

    sop_instance_uid = pydicom.uid.generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceNumber = str(instance_index_1based)

    ds.PixelData = pixels_2d.tobytes()
    ds.save_as(str(save_path))


def _arr_zyx_from_xyz(vol_xyz: np.ndarray) -> np.ndarray:
    """Convert (X, Y, Z) -> (Z, Y, X) slices."""
    return np.transpose(vol_xyz, (2, 1, 0))


# =============================================================================
# Series export functions
# =============================================================================

def export_ray_orientation(
    meta: NiftiSeriesMeta,
    out_root: Path,
    series_prefix: str,
    modality: str = "CT",
    series_description: Optional[str] = None,
) -> Path:
    """
    Export DICOM series in RayStation-friendly orientation:
      arr_hfs = rot90(flip(arr_zyx, axis=1), k=2, axes=(1,2))
      IOP = [1,0,0,0,1,0], PatientPosition="HFS"
      IPP_k = IPP_first - k*SliceThickness
    
    Args:
        series_description: Custom SeriesDescription (if None, uses series_prefix)
    
    Returns:
        out_dir: Path to the output directory containing DICOM slices
    """
    arr_zyx = _arr_zyx_from_xyz(meta.vol_xyz)
    arr_hfs = np.rot90(np.flip(arr_zyx, axis=1), k=2, axes=(1, 2))

    series_uid = pydicom.uid.generate_uid()
    series_number = str(random.randint(0, 9999))

    out_dir = out_root / f"{series_prefix}_rayOrientation_slices"
    out_dir.mkdir(parents=True, exist_ok=True)

    iop = [1, 0, 0, 0, 1, 0]
    ipp0 = meta.IPP_first
    thk = float(meta.SliceThickness)

    # Determine file prefix based on modality
    file_prefix = "MR" if modality == "MR" else "CT"
    
    # Use custom series description or default to series_prefix
    desc = series_description if series_description else series_prefix

    for k in range(arr_hfs.shape[0]):
        slice_arr = arr_hfs[k, :, :]
        ipp_k = [ipp0[0], ipp0[1], ipp0[2] - k * thk]
        fn = out_dir / f"{file_prefix}_{(k+1):04d}.dcm"
        ds = _base_ds(meta, str(fn), modality=modality)
        _write_instance(
            ds=ds,
            pixels_2d=slice_arr,
            series_uid=series_uid,
            series_number=series_number,
            series_description=desc,
            iop=iop,
            ipp=ipp_k,
            pixel_spacing=meta.PixelSpacing,
            slice_thickness=meta.SliceThickness,
            patient_position="HFS",
            instance_index_1based=k + 1,
            save_path=fn,
        )

    return out_dir


# =============================================================================
# Inference function
# =============================================================================

@torch.no_grad()
def run_inference(
    cfg_path: str,
    ckpt_path: str,
    mri_np: np.ndarray,
    device_override: Optional[str] = None,
    amp: bool = False,
    clamp_hu: bool = True,
) -> np.ndarray:
    """
    Run inference on a pre-loaded MR volume.
    
    Args:
        cfg_path: Path to model config YAML
        ckpt_path: Path to model checkpoint
        mri_np: (D, H, W) MR volume as numpy array
        device_override: Device override (cuda/cpu)
        amp: Use automatic mixed precision
        clamp_hu: Clamp output to HU range
    
    Returns:
        y_hu: (D, H, W) predicted CT volume in HU
    """
    cfg = load_config(cfg_path)
    device = choose_device(device_override or cfg.train.device_preference)

    exp_shape = tuple(cfg.data.enforce_shape)
    if tuple(mri_np.shape) != exp_shape:
        raise ValueError(f"MR shape {mri_np.shape} != expected {exp_shape}. Please pre-align to training shape.")

    # Scale input same as training
    x_np = scale_input(mri_np, cfg.data.scale_input)
    x_t = torch.from_numpy(x_np[None, None, ...]).to(device=device, dtype=torch.float32)

    # Model
    model = UNet3D(**cfg.model.__dict__).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.eval()

    # Inference (whole volume)
    if amp and device.type == "cuda":
        from torch.amp import autocast
        with autocast(device_type="cuda", enabled=True):
            pred = model(x_t)
    else:
        pred = model(x_t)

    y_pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()

    # Invert target scaling to HU if needed
    y_hu = invert_target_scaling(
        y_pred,
        mode=cfg.data.scale_target,
        hu_min=float(cfg.data.hu_min),
        hu_max=float(cfg.data.hu_max),
    )
    if cfg.data.scale_target == "hu" and clamp_hu:
        y_hu = np.clip(y_hu, float(cfg.data.hu_min), float(cfg.data.hu_max))

    return y_hu


# =============================================================================
# Main workflow
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete MRI-Only Workflow: Inference + DICOM Export (sCT, GT, MR)"
    )

    # Model / Inference Args
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--mr", required=True, type=Path, help="Input MR NIfTI")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth CT NIfTI")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    # Reference DICOM
    parser.add_argument("--ref", required=True, type=Path, help="Reference CT DICOM (geometry source)")

    # Alignment
    parser.add_argument(
        "--align", action="store_true",
        help="Align MR and sCT to GT CT using rigid registration (requires SimpleITK)"
    )
    parser.add_argument(
        "--align-iterations", type=int, default=500,
        help="Max iterations for alignment registration (default: 500)"
    )

    # Output
    parser.add_argument("--out", required=True, type=Path, help="Output root directory")

    args = parser.parse_args()

    # Validate inputs
    for p, name in [(args.mr, "MR"), (args.gt, "GT"), (args.ref, "Reference DICOM")]:
        if not p.exists():
            print(f"Error: {name} file not found: {p}")
            sys.exit(1)

    # Setup output directories
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    nifti_dir = out_root / "nifti_intermediate"
    nifti_dir.mkdir(exist_ok=True)

    dicom_dir = out_root / "dicom"
    dicom_dir.mkdir(exist_ok=True)

    # Extract patient ID from MR filename
    patient_id = extract_mouse_id(args.mr.name)
    print(f"Patient ID: {patient_id}")

    # Generate shared UIDs for all series
    study_uid = pydicom.uid.generate_uid()
    frame_of_reference_uid = pydicom.uid.generate_uid()
    print(f"Study Instance UID: {study_uid}")
    print(f"Frame of Reference UID: {frame_of_reference_uid}")

    # =========================================================================
    # Step 1: Run Inference (MR -> sCT)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Running Inference (MR -> sCT)")
    print("=" * 60)

    # Load MR volume
    mri_np, mri_affine, mri_header = load_nifti(args.mr, reorient_to_canonical=False)
    print(f"  MR shape: {mri_np.shape}")

    # Run inference
    sct_np = run_inference(
        cfg_path=args.config,
        ckpt_path=args.ckpt,
        mri_np=mri_np,
        device_override=args.device,
    )
    print(f"  sCT shape: {sct_np.shape}")
    print(f"  sCT HU range: {sct_np.min():.1f} to {sct_np.max():.1f}")

    # Save intermediate sCT NIfTI
    sct_nifti_path = nifti_dir / "sCT_prediction.nii.gz"
    save_nifti(sct_nifti_path, sct_np, affine=mri_affine, header=mri_header)
    print(f"  Saved intermediate sCT to: {sct_nifti_path}")

    # Paths for export (will be updated if alignment is enabled)
    mr_export_path = args.mr
    sct_export_path = sct_nifti_path
    gt_export_path = args.gt

    # =========================================================================
    # Step 1.5: Align MR and sCT to GT CT (optional)
    # =========================================================================
    if args.align:
        print("\n" + "=" * 60)
        print("Step 1.5: Rigid Alignment to GT CT")
        print("=" * 60)
        
        if not SITK_AVAILABLE:
            print("  ERROR: SimpleITK not available. Install with: pip install SimpleITK")
            print("  Skipping alignment...")
        else:
            aligned_dir = nifti_dir / "aligned"
            aligned_dir.mkdir(exist_ok=True)
            
            # Align MR to GT CT
            print("\n  Aligning MR to GT CT...")
            mr_aligned_path = aligned_dir / "MR_aligned_to_GT.nii.gz"
            try:
                mr_aligned_np, mr_transform = rigid_align(
                    fixed_path=args.gt,
                    moving_path=args.mr,
                    output_path=mr_aligned_path,
                    max_iterations=args.align_iterations,
                    verbose=True,
                )
                mr_export_path = mr_aligned_path
                print(f"    Saved: {mr_aligned_path}")
            except Exception as e:
                print(f"    MR alignment failed: {e}")
            
            # Align sCT to GT CT
            print("\n  Aligning sCT to GT CT...")
            sct_aligned_path = aligned_dir / "sCT_aligned_to_GT.nii.gz"
            try:
                sct_aligned_np, sct_transform = rigid_align(
                    fixed_path=args.gt,
                    moving_path=sct_nifti_path,
                    output_path=sct_aligned_path,
                    max_iterations=args.align_iterations,
                    verbose=True,
                )
                sct_export_path = sct_aligned_path
                print(f"    Saved: {sct_aligned_path}")
            except Exception as e:
                print(f"    sCT alignment failed: {e}")
            
            print("\n  Alignment complete. Using aligned volumes for DICOM export.")

    # =========================================================================
    # Step 2: Export DICOM Series (sCT, GT, MR)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Exporting DICOM Series")
    print("=" * 60)

    # Export sCT (aligned if --align was used)
    print("\n  Exporting sCT series...")
    sct_meta = NiftiSeriesMeta(
        nifti_path=sct_export_path,
        reference_dicom=args.ref,
        patient_id=patient_id,
        study_uid=study_uid,
        frame_of_reference_uid=frame_of_reference_uid,
    )
    sct_dir = export_ray_orientation(
        sct_meta, dicom_dir, series_prefix="sCT", modality="CT",
        series_description="Synthetic CT"
    )
    print(f"    sCT DICOM series: {sct_dir}")

    # Export GT (always the original - it's the reference)
    print("\n  Exporting GT series...")
    gt_meta = NiftiSeriesMeta(
        nifti_path=gt_export_path,
        reference_dicom=args.ref,
        patient_id=patient_id,
        study_uid=study_uid,
        frame_of_reference_uid=frame_of_reference_uid,
    )
    gt_dir = export_ray_orientation(
        gt_meta, dicom_dir, series_prefix="GT", modality="CT",
        series_description="Ground truth CT"
    )
    print(f"    GT DICOM series: {gt_dir}")

    # Export MR (aligned if --align was used)
    print("\n  Exporting MR series...")
    mr_meta = NiftiSeriesMeta(
        nifti_path=mr_export_path,
        reference_dicom=args.ref,
        patient_id=patient_id,
        study_uid=study_uid,
        frame_of_reference_uid=frame_of_reference_uid,
        is_mr=True,  # Enable MR-specific scaling and tags
    )
    mr_dir = export_ray_orientation(
        mr_meta, dicom_dir, series_prefix="MR", modality="MR",
        series_description="MR"
    )
    print(f"    MR DICOM series: {mr_dir}")

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {out_root}")
    print(f"  - NIfTI intermediate: {nifti_dir}")
    print(f"  - DICOM series: {dicom_dir}")


if __name__ == "__main__":
    main()
