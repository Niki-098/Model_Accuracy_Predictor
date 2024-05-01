// upload_script.js

// JavaScript code to handle form submission
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Get the input element for file upload
    var fileInput = document.getElementById('file-input');
    // Get the selected file
    var file = fileInput.files[0];

    // Create a FormData object to send file data with the request
    var formData = new FormData();
    formData.append('file', file);

    // Send a POST request to the server to handle file upload
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload/'); // Replace '/upload/' with the URL to handle file upload
    xhr.send(formData);

    // Optionally, handle the response from the server
    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                // File upload successful
                alert('File uploaded successfully!');
            } else {
                // File upload failed
                alert('File upload failed. Please try again.');
            }
        }
    };
});
