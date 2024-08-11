## AIFinalProject README

### Project Title: AIFinalProject

---

### Project Description
This project focuses on implementing AI techniques to predict a person's next menstrual cycle. It includes data preprocessing, model training, evaluation, and visualization of results.

---

### Folder Structure
The project repository is structured as follows:
```
AIFinalProject/
│
├── predictor.py             # Main Python script for prediction
├── Datasets/                # Folder containing datasets used in the project
├── Models/                  # Folder containing trained models
├── Static/                  # Folder containing static files (e.g., images)
├── Templates/               # Folder containing template files
├── README.md                # README file (this document)

```

---

### Installation
To set up the project on your local machine, follow these steps:

1. **Clone the repository:**
   ```
   git clone <repository_url>
   cd AIFinalProject
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

---

### Usage


- **Running the predictor script:**
  ```
  python predictor.py
  ```
---

### Dataset
The `Datasets` folder contains the datasets used in this project. Please ensure that the datasets are in the correct format and properly preprocessed before use.

---

### Models
The `Models` folder contains the pre-trained models. You can load these models for prediction or further training.

---

### Results
Results of the model predictions and evaluations will be stored in the `Results` folder. You can find detailed performance metrics and visualizations in this directory.

---
### Loading the Webpage Locally
To load the webpage locally, follow these steps:

Ensure you are in the project directory:

```
cd AIFinalProject
```
Run the Flask application:
```
export FLASK_APP=predictor.py     # On Windows, use `set FLASK_APP=predictor.py`
flask run
```
Access the webpage:
Open your web browser and go to http://127.0.0.1:5000/ to view the webpage locally.
---