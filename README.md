## IntelliProfiler: A research workflow for multi-animal behavioral analysis using R and Python

This repository provides the source code for **IntelliProfiler**, a research workflow consisting of R and Python scripts for analyzing locomotor activity and social proximity in group-housed mice. IntelliProfiler processes positional data acquired from a commercially available high-resolution RFID floor plate (eeeHive2D, Phenovance LLC, Japan; Lipp et al., 2024) and is not validated with other hardware configurations. A workflow generates automated, quantitative behavioral metrics and visualizations.

> ⚠️ **Note**: IntelliProfiler is a research workflow. It does not include or control any RFID hardware components. Users must obtain compatible RFID tracking systems (e.g., eeeHive2D) separately.

**Details of the IntelliProfiler workflow are described in the following article:**  
[_IntelliProfiler: a research workflow for analyzing multiple animals with a high-resolution home-cage RFID system_]([(https://www.nature.com/articles/s41684-025-01668-4)])  
🧾 Lab Animal (2026)

---

### Getting Started

Follow the steps below to run IntelliProfiler and analyze RFID tracking data from group-housed mice.

---

### Prerequisites

The following R packages are required to run the scripts:

- `tidyverse`
- `openxlsx`
- `lubridate`

You can install these packages using the following commands:

```r
install.packages("tidyverse")
install.packages("openxlsx")
install.packages("lubridate")
```

Alternatively, the code will attempt to install missing packages when executed.

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/ShoheiOchi/IntelliProfiler.git
   ```

2. Open R or RStudio and set your working directory to the cloned repository:
   ```r
   setwd("path_to_repository/IntelliProfiler")
   ```

3. Ensure the necessary R packages are installed (the script will check for packages and install them if needed).

### Running the Analysis

1. **Data Preparation**: Ensure that RFID tracking data is logged using TeraTerm software and saved as a .txt file. A sample file Test.txt is provided in the data/ directory. You can replace this file with your own .txt data for real experiments.

2. **Run the Main Script**: 
   ```r
   source("scripts/IP_general.R")

3. **File Input**:
   When prompted, select the .txt file that contains the RFID tracking data of the mice.

4. **Output**:
   The script will process the data and generate the following outputs:
   - Excel files containing the processed data for each mouse.
   - 2D plots of the tracked positions for each mouse in PDF format.
   - Time series plots of the X and Y positions.
   - Distance plots comparing the social distances between mice.

### Data Directory Structure

- **scripts/**: Contains the main R script (IP_general.R) and optional Python analysis scripts.
- **data/**: Contains sample input data  (you can place your `.txt` data files here).
- **results/**: Output files (Excel, PDF plots) will be saved here after analysis.

5. **Note**:
   The raw log file (.txt) exported from TeraTerm is parsed by `IP_general.R`, which reformats the data into per-second positions, handles missing values by simple interpolation, and prepares the dataset for downstream analyses.

### Citation

If you use IntelliProfiler in your research, please cite:
Ochi S, Inada H, Osumi N. IntelliProfiler: a research workflow for analyzing multiple animals with a high-resolution home-cage RFID system. Lab Animal. 2026;55:48–63. https://doi.org/10.1038/s41684-025-01668-4

### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
