import React, { useState } from 'react';
import './imageDrop.css';

function ImageDrop() {
    const [image1, setImage1] = useState(null);
    const [image2, setImage2] = useState(null);

    const allowDrop = (event) => {
        event.preventDefault();
    };

    const drop = (event, setImage) => {
        event.preventDefault();
        let dt = event.dataTransfer;
        let file = dt.files[0];

        // Update local state with image for preview
        const reader = new FileReader();
        reader.onload = () => {
            setImage(reader.result);
        };
        reader.readAsDataURL(file);

        // Send the file to the backend
        sendImageToServer(file);
    };

    const sendImageToServer = (file) => {
        const url = 'http://127.0.0.1:5000/upload';  // URL of your Flask endpoint
        const formData = new FormData();
        formData.append('file', file);

        fetch(url, {
            method: 'POST',
            body: formData,
        })
        .then(response => response.blob())
        .then(imageBlob => {
            // Convert the Blob into an Image and update state to show the processed image
            const imageObjectURL = URL.createObjectURL(imageBlob);
            setImage1(imageObjectURL);  // Assuming you want to show the processed image
        })
        .catch(error => console.error('Error:', error));
    };

    return (
        <div className="image-drop-container">
            <div className="drop-area" onDragOver={allowDrop} onDrop={(event) => drop(event, setImage1)}>
                {image1 ? <img src={image1} alt="Dropped Image" /> : 'Drop the first image here'}
            </div>
            <div className="drop-area" onDragOver={allowDrop} onDrop={(event) => drop(event, setImage2)}>
                {image2 ? <img src={image2} alt="Dropped Image" /> : 'Drop the second image here'}
            </div>
        </div>
    );
}

export default ImageDrop;
