# Mars Surface AI

A Streamlit web application for classifying Mars surface images as **Rocky** or **Smooth**.

## Features
- Upload a trained Random Forest `.joblib` model.
- Upload one or more JPG, JPEG, or PNG images.
- Extract image features with SqueezeNet.
- Predict Rocky/Smooth and display model confidence scores.
- Download prediction results as CSV.

## Libraries and technologies

The app uses Streamlit, Pandas, NumPy, Scikit-learn, Joblib, Pillow, PyTorch, and Torchvision. It also uses Python's built-in `io` module for in-memory image handling.

See **[LIBRARIES.md](LIBRARIES.md)** for what each library does and how the training and prediction pipelines fit together.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file to `app.py`.
5. Deploy. Streamlit will install the packages listed in `requirements.txt`.
6. Open the deployed app URL.
7. In the **Upload Images & Predict** tab, upload your trained `mars_orange_random_forest.joblib` model, then upload Mars JPG/PNG images.

## Local run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Model file

The trained `.joblib` model is intentionally not included in this repository. Upload it through the app's model uploader after deployment. Do not commit private datasets or model files unless you intentionally want them public.

## Critical model compatibility check

The app currently uses `torchvision` SqueezeNet 1.1 with ImageNet weights and torchvision's standard inference preprocessing. Orange's Image Embedding widget may use a different SqueezeNet implementation, preprocessing, weight version, or output layer.

The fact that both produce 1,000 values does **not** prove their embeddings are equivalent. Before relying on predictions, embed the same image in Orange and this app and compare the resulting feature vectors. If they differ materially, the website's image embedding pipeline must be adjusted to match Orange exactly.

## Dataset format for training

The training interface expects a CSV with:
- `category` column containing `rocky` and `smooth`
- Numeric embedding columns `n0` through `n999`

The direct-image prediction interface does not require users to upload an embeddings CSV.
