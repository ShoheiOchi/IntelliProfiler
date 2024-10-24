## IntelliProfiler: A Novel Tool for Multi-Animal Behavioral Analysis

This repository contains the R and Python code used in the study titled *"IntelliProfiler: A Novel Analytic Tool for Behavior Dynamics of Multiple Animals in a Home Cage."* The IntelliProfiler system enables high-resolution tracking and behavioral analysis of multiple mice equipped with RFID transponders in their natural home cage environment. The system facilitates automated analysis of locomotor activity, social interactions, and spatial positioning, making it a valuable tool in neuroscience and behavioral science research.


### Getting Started

Follow the instructions below to set up and run IntelliProfiler to track the position and behavior of mice equipped with RFID transponders.


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
   git clone https://github.com/<your_username>/IntelliProfiler.git
   ```

2. Open R or RStudio and set your working directory to the cloned repository:
   ```r
   setwd("path_to_repository/IntelliProfiler")
   ```

3. Ensure the necessary R packages are installed (the script will check for packages and install them if needed).

### Running the Analysis

1. **Data Preparation**: Ensure that RFID tracking data is logged using TeraTerm software and saved as a `.txt` file. We have provided a sample file `Test.txt` in the `data/` directory. This file contains RFID tracking data that can be used to test the IntelliProfiler system. You can replace this file with your own `.txt` data for real experiments.

2. **Run the Main Script**: 
   ```r
   source("scripts/IntelliProfiler.R")

3. **File Input**:
   When prompted, select the `.txt` file that contains the RFID tracking data of the mice.

4. **Output**:
   The script will process the data and generate the following outputs:
   - Excel files containing the processed data for each mouse.
   - 2D plots of the tracked positions for each mouse in PDF format.
   - Time series plots of the X and Y positions.
   - Distance plots comparing the social distances between mice.

### Data Directory Structure

- **scripts/**: Contains the main R scripts for data processing and analysis.
- **data/**: Contains sample input data (you can place your `.txt` data files here).
- **results/**: After running the analysis, this directory will contain the output files including Excel data and PDF plots.

### Future Citation

Once published, please use the appropriate citation to reference IntelliProfiler in your research. This section will be updated with citation details upon acceptance of the manuscript.

### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
