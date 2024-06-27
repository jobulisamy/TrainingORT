import * as ort from 'onnxruntime-web'

const ort = ortTraining;
const modelURL = 't1train.onnx'; 

let trainingModel;
let optimizer;
let loss;
let trainer;
let epochs = 10;

async function startTraining() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    if (files.length === 0) {
        alert('Please upload some images.');
        return;
    }

    const progressDiv = document.getElementById('progress');
    const outputDiv = document.getElementById('output');
    outputDiv.innerHTML = '';

    // Load the ONNX model
    const onnxModel = await fetch(modelURL).then(response => response.arrayBuffer());
    const onnxModelBytes = new Uint8Array(onnxModel);

    // Load ONNX model into ONNX Runtime
    trainingModel = await ort.InferenceSession.create(onnxModelBytes);
    
    // Define the optimizer
    optimizer = new ort.AdamWOptimizer({ learningRate: 0.001 });

    // Define the loss function
    loss = new ort.CrossEntropyLoss();

    // Define the model description
    const modelDesc = {
        inputs: [
            { name: 'input', shape: [1, 3, 32, 32], type: 'float32' }
        ],
        outputs: [
            { name: 'output', shape: [1, 2], type: 'float32' }
        ]
    };

    // Create the trainer
    trainer = new ort.ORTTrainer(trainingModel, modelDesc, optimizer, loss);

    // Preprocess images and labels
    const { images, labels } = await preprocessImages(files);

    // Start the training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
        let epochLoss = 0;
        for (let i = 0; i < images.length; i++) {
            const inputs = { 'input': new Float32Array(images[i]) };
            const targets = { 'output': new Float32Array(labels[i]) };
            const lossValue = await trainer.trainStep(inputs, targets);
            epochLoss += lossValue;
        }
        epochLoss /= images.length;
        outputDiv.innerHTML += `Epoch ${epoch + 1}/${epochs}, Loss: ${epochLoss.toFixed(4)}\n`;
    }
    progressDiv.innerHTML = 'Training Complete';
}

async function preprocessImages(files) {
    const images = [];
    const labels = [];

    for (const file of files) {
        const img = await loadImage(file);
        const resizedImage = resizeImage(img, 32, 32);
        const imageArray = imageToFloat32Array(resizedImage);

        images.push(imageArray);
        // Assuming the folder name or some part of the file name indicates the label
        // This needs to be adjusted according to how the labels are provided
        labels.push(file.name.toLowerCase().includes('cat') ? [1, 0] : [0, 1]);
    }

    return { images, labels };
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        const reader = new FileReader();
        reader.onload = e => {
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function resizeImage(image, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, width, height);
    return canvas;
}

function imageToFloat32Array(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const float32Array = new Float32Array(canvas.width * canvas.height * 3);
    for (let i = 0; i < data.length; i += 4) {
        const j = (i / 4) * 3;
        float32Array[j] = data[i] / 255;
        float32Array[j + 1] = data[i + 1] / 255;
        float32Array[j + 2] = data[i + 2] / 255;
    }
    return float32Array;
}
