from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    ApertureStats,
    aperture_photometry,
)


# ------------------------------------------------------------
# 1. Locate the FITS file
# ------------------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent

filename = (
    project_dir
    / "lib"
    / "raw fits"
    / "lights"
    / "2026-08-24_21-20-12_Ha_-10.00_300.00s_0000.fits"
)

print("Looking for:", filename)
print("File exists:", filename.exists())

if not filename.exists():
    raise FileNotFoundError(f"FITS file was not found:\n{filename}")


# ------------------------------------------------------------
# 2. Read the image from the primary HDU
# ------------------------------------------------------------

with fits.open(filename) as hdul:
    hdul.info()
    data = hdul[0].data

if data is None:
    raise ValueError("The primary HDU does not contain image data.")

if data.ndim != 2:
    raise ValueError(f"Expected a 2D image, but got shape {data.shape}")

# Convert before doing calculations
data = np.asarray(data, dtype=np.float64)

print("Image shape:", data.shape)
print("Data type:", data.dtype)
print("Minimum pixel value:", np.nanmin(data))
print("Maximum pixel value:", np.nanmax(data))


# ------------------------------------------------------------
# 3. Estimate and subtract the global background
# ------------------------------------------------------------

mean, median, std = sigma_clipped_stats(
    data,
    sigma=3.0
)

print("Background mean:", mean)
print("Background median:", median)
print("Background standard deviation:", std)

# DAOStarFinder works better with the background removed
data_subtracted = data - median


# ------------------------------------------------------------
# 4. Find stars
# ------------------------------------------------------------

finder = DAOStarFinder(
    threshold=5.0 * std,
    fwhm=3.0,
    exclude_border=True
)

sources = finder(data_subtracted)


# ------------------------------------------------------------
# 5. Create apertures and measure the stars
# ------------------------------------------------------------

if sources is None:
    print("No stars were detected.")

else:
    print(f"Number of detected stars: {len(sources)}")
    print()
    print(sources)

    # Photutils uses (x, y) coordinates:
    # x = image column
    # y = image row
    positions = np.column_stack(
        (
            sources["x_centroid"],
            sources["y_centroid"],
        )
    )

    # A circular aperture centered on every detected star
    star_apertures = CircularAperture(
        positions,
        r=3.1
    )

    # An annulus around every star for estimating local background
    background_apertures = CircularAnnulus(
        positions,
        r_in=6.0,
        r_out=9.0
    )

    # Measure the raw flux inside each stellar aperture
    phot_table = aperture_photometry(
        data,
        star_apertures
    )

    # Estimate the local background in each annulus
    background_stats = ApertureStats(
        data,
        background_apertures,
        sigma_clip=None
    )

    # Median background per pixel for each star
    local_background = background_stats.median

    # Area of each stellar aperture in pixels
    aperture_area = star_apertures.area

    # Total background inside each stellar aperture
    total_background = local_background * aperture_area

    # Background-subtracted aperture flux
    flux_background_subtracted = (
        phot_table["aperture_sum"] - total_background
    )

    # Add useful columns to the output table
    phot_table["source_id"] = sources["id"]
    phot_table["x_centroid"] = sources["x_centroid"]
    phot_table["y_centroid"] = sources["y_centroid"]
    phot_table["finder_flux"] = sources["flux"]
    phot_table["peak"] = sources["peak"]
    phot_table["local_background"] = local_background
    phot_table["total_background"] = total_background
    phot_table["flux_background_subtracted"] = (
        flux_background_subtracted
    )

    print()
    print("Aperture photometry results:")
    print(phot_table)


    # --------------------------------------------------------
    # 6. Print one readable line per star
    # --------------------------------------------------------

    print()
    print("Detected stars:")

    for row in phot_table:
        print(
            f"Star {int(row['source_id'])}: "
            f"x={row['x_centroid']:.2f}, "
            f"y={row['y_centroid']:.2f}, "
            f"flux={row['flux_background_subtracted']:.2f}"
        )


    # --------------------------------------------------------
    # 7. Save the results
    # --------------------------------------------------------

    output_file = project_dir / "detected_stars.ecsv"

    phot_table.write(
        output_file,
        format="ascii.ecsv",
        overwrite=True
    )

    print()
    print("Saved photometry table to:", output_file)


    # --------------------------------------------------------
    # 8. Plot the image and apertures
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 8))

    # Use a reasonable display range
    display_min = median
    display_max = median + 10.0 * std

    ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=display_min,
        vmax=display_max
    )

    # Plot stellar apertures in red
    star_apertures.plot(
        ax=ax,
        color="red",
        lw=1.0
    )

    # Plot background annuli in cyan
    background_apertures.plot(
        ax=ax,
        color="cyan",
        lw=1.0
    )

    ax.set_title(
        f"Detected stars: {len(sources)}"
    )
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    plt.tight_layout()
    plt.show()