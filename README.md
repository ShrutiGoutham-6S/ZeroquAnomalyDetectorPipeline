---

# Zeroqu Anomaly Detector Pipeline

## Setup & Run Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ZeroquAnomalyDetectorPipeline.git
   cd ZeroquAnomalyDetectorPipeline
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add the dataset**

   * Place your `SOL_RELIANCE.xlsx` file in the project root folder.

4. **Train the models**

   ```bash
   python model_trainer.py
   ```

   → This will generate trained model and metadata files in the `/models` folder.

5. **Run the integrated anomaly detection pipeline**

   ```bash
   python integ.py
   ```

   → This will load the trained models, detect anomalies for the test sample, and print a JSON summary.
---
