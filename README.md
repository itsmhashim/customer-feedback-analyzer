# Sentiment Analysis API

This repository contains a sentiment analysis API built using FastAPI and Hugging Face's pre-trained RoBERTa model. The API predicts whether a given text is **Positive**, **Negative**, or **Neutral**.

---

## Features
- Predict sentiment for text inputs.
- Handles text preprocessing, including:
  - Removing special characters.
  - Spell correction using TextBlob.
  - Validation for empty inputs and overly long texts.
- Implements a secure and efficient FastAPI backend.

---

## Requirements

Ensure you have Python 3.8 or higher installed. Install the dependencies using the `requirements.txt` file:

```bash
  pip install -r requirements.txt
```

### Dependencies
- `fastapi`
- `uvicorn`
- `torch`
- `transformers`
- `textblob`
- `pydantic`

---

## File Structure

```plaintext
.
├── app.py                 # Main FastAPI application
├── requirements.txt
└── README.md         
└── customer-feedback-analyzer.ipynb
```

---


## Dataset
The model was trained on the **[McDonald's Store Reviews Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)**. Users can download the dataset from the source and use it for further experimentation.


---


## How the Model is Loaded

The model files are hosted on Hugging Face Model Hub. During runtime, the API fetches the model and tokenizer directly from Hugging Face. This eliminates the need to store large model files in the repository.

Ensure that the Hugging Face `transformers` library is installed and properly configured.

---


## Preprocessing and Training

The Jupyter notebook included in this repository (`customer-feedback-analyzer.ipynb`) outlines the entire preprocessing and training process for the sentiment analysis model. This includes:

1. **Dataset Preparation:**
   - Cleaning and preprocessing the text data.
   - Splitting the dataset into training, validation, and test sets.

2. **Model Training:**
   - Fine-tuning the pre-trained RoBERTa model on the processed dataset.

3. **Evaluation:**
   - Calculating accuracy and loss on the test dataset.

Feel free to explore the notebook for detailed steps and code.

---

## How to Run the API

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Start the API**
   Run the following command in the project directory:
   ```bash
   uvicorn app:app --reload
   ```

3. **Access the API**
   The API will be accessible at:
   ```
   http://127.0.0.1:8000
   ```

4. **Test the API**
   Use tools like Postman, cURL, or a browser to send POST requests to:
   ```
   http://127.0.0.1:8000/predict
   ```

   Example Request Body:
   ```json
   {
       "text": "The product was amazing!"
   }
   ```

---

## Demo

Access the deployed service here: [Sentiment Analysis API Demo](https://customer-feedback-analyzer.onrender.com) 


---

## Future Improvements
- Add sarcasm detection.
- Integrate topic modeling.
- Enhance spell correction with a custom-trained model.


---

Feel free to contribute to this project by submitting issues or pull requests!
