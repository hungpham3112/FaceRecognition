import cv2
import numpy as np
import streamlit as st
from lib import eigenfaces, recognize, detect_and_crop_face


def main():
    st.set_page_config(page_title="Eigenfaces Face Recognition", layout="wide")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_files = st.file_uploader(
            "Upload TRAINING images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        labels_input = st.text_input(
            """Enter the names of the people in the TRAINING images (comma separated,
                                     and reversed). For example: you uploaded 3 images: image1, image2, image3 to
                                     streamlit then the label is image3, ima  ge2, image1"""
        )

    if uploaded_files and labels_input:
        labels = labels_input.split(",")
        if len(uploaded_files) != len(labels):
            st.error("The number of labels should match the number of uploaded images.")
            return

        train_images = []
        for uploaded_file in uploaded_files:
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
            cropped_face = detect_and_crop_face(img)
            if cropped_face is None:
                st.error(
                    "Face not detected in one of the training images. Please ensure clear visibility of faces."
                )
                return
            train_images.append(cropped_face)

        mean_face, eigenvectors = eigenfaces(train_images)

        with col2:
            uploaded_file = st.file_uploader(
                "Now, upload an image for RECOGNITION", type=["jpg", "jpeg", "png"]
            )

            if uploaded_file:
                img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
                cropped_face = detect_and_crop_face(img)
                if cropped_face is None:
                    st.error(
                        "Face not detected in the recognition image. Please ensure clear visibility of faces."
                    )
                    return
                recognized_label = recognize(
                    cropped_face, train_images, mean_face, eigenvectors, labels
                )
                st.text(f"This is a face of {recognized_label}.")
                st.image(img, channels="BGR", use_column_width=True)


if __name__ == "__main__":
    main()
