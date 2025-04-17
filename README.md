## IntelliProfiler: A software tool for multi-animal behavioral analysis using R and Python

This repository provides the source code for **IntelliProfiler**, a software tool for analyzing locomotor activity and social proximity in group-housed mice. Developed in R and Python, IntelliProfiler processes positional data acquired from an external high-resolution home-cage RFID system (eeeHive2D, Lipp et al., 2024) and generates automated, quantitative behavioral metrics and visualizations.

> ⚠️ **Note**: IntelliProfiler is a software pipeline. It does not include any hardware components such as RFID sensor arrays. Users must obtain compatible RFID tracking systems (e.g., eeeHive2D) separately.

**Details of the IntelliProfiler tool are described in the following preprint:**  
[_IntelliProfiler: a novel software pipeline for analyzing multiple animals with a high-resolution home-cage RFID system_](https://www.biorxiv.org/content/10.1101/2024.10.23.619967v2)  
🧾 bioRxiv Preprint (2025)

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

### Future Citation

Once published, please cite:
Ochi S, Inada H, Osumi N. IntelliProfiler: a novel software pipeline for analyzing multiple animals with a high-resolution home-cage RFID system. bioRxiv. 2025. [https://doi.org/10.1101/2024.10.23.619967](https://www.biorxiv.org/content/10.1101/2024.10.23.619967v2)

### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
