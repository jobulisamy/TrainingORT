import * as ort from 'onnxruntime-web'

async function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.src = reader.result;
            img.onload = () => resolve(img);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function preprocessImage(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 32, 32);
    const imageData = ctx.getImageData(0, 0, 32, 32);
    const data = new Float32Array(3 * 32 * 32);
    for (let i = 0; i < 32 * 32; i++) {
        data[i] = imageData.data[i * 4] / 255;        // R
        data[i + 32 * 32] = imageData.data[i * 4 + 1] / 255;  // G
        data[i + 2 * 32 * 32] = imageData.data[i * 4 + 2] / 255;  // B
    }
    return data;
}

function addLabelInput(file, index) {
    const labelContainer = document.getElementById('label-container');
    const labelDiv = document.createElement('div');
    labelDiv.className = 'label-div';

    const imgLabel = document.createElement('label');
    imgLabel.innerHTML = `Image ${index + 1} (${file.name}): `;
    labelDiv.appendChild(imgLabel);

    const select = document.createElement('select');
    select.id = `label-${index}`;
    const catOption = document.createElement('option');
    catOption.value = 0;
    catOption.text = 'Cat';
    const dogOption = document.createElement('option');
    dogOption.value = 1;
    dogOption.text = 'Dog';
    select.appendChild(catOption);
    select.appendChild(dogOption);
    labelDiv.appendChild(select);

    labelContainer.appendChild(labelDiv);
}

document.getElementById('file-upload').addEventListener('change', async (event) => {
    const files = event.target.files;
    const labelContainer = document.getElementById('label-container');
    labelContainer.innerHTML = '';  // Clear previous inputs
    for (let i = 0; i < files.length; i++) {
        addLabelInput(files[i], i);
    }
});

 async function runTraining() {
    const fileInput = document.getElementById('file-upload');
    const files = fileInput.files;
    if (files.length === 0) {
        alert("Please upload images for training.");
        return;
    }

    const tensors = [];
    const labels = new Float32Array(files.length);
    for (let i = 0; i < files.length; i++) {
        const img = await loadImage(files[i]);
        const data = preprocessImage(img);
        const tensor = new ort.Tensor('float32', data, [1, 3, 32, 32]);
        tensors.push(tensor);

        const labelSelect = document.getElementById(`label-${i}`);
        labels[i] = parseInt(labelSelect.value);
    }

    // Load the ONNX training artifacts
    const trainingModel = await ort.TrainingSession.create('training_model.onnx');
    const optimizerModel = await ort.TrainingSession.create('optimizer_model.onnx');
    const evalModel = await ort.TrainingSession.create('eval_model.onnx');
    const checkpoint = await ort.Checkpoint.load('checkpoint_file.onnx');

    // Initialize optimizer and training session
    const optimizer = new ort.Optimizer(optimizerModel, 'AdamW', { learningRate: 0.01 });
    const trainingSession = new ort.TrainingSession(trainingModel, optimizer, checkpoint);

    
    function computeLoss(outputs, labels) {
        let loss = 0;
        for (let i = 0; i < outputs[0].data.length; i++) {
            loss += (outputs[0].data[i] - labels[i]) ** 2;
        }
        return loss / outputs[0].data.length;
    }

    for (let epoch = 0; epoch < 10; epoch++) {
        for (let i = 0; i < tensors.length; i++) {
            const input = tensors[i];
            const label = labels[i];

            // Run forward pass
            const outputs = await trainingSession.run({ input });

            // Compute loss
            const loss = computeLoss(outputs, label);

            // Run backward pass and update weights
            await trainingSession.backward(loss);
            optimizer.step();
            trainingSession.lazyResetGrad();

            console.log(`Epoch ${epoch}, Step ${i}: Loss = ${loss}`);
        }
    }

    console.log('Training complete');
}
