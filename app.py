from flask import Flask, request, render_template, redirect, url_for
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import io
import uuid

# Define a transformation (e.g., resizing and converting to tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ConvBlock definition
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ImageClassificationBase definition
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

# ResNet9 definition
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'model/plant-disease-model-complete.pth'
model = torch.load(model_path, map_location=device)
model.eval()

# Define the Flask app
app = Flask(__name__)

# Define the class names (adjust this according to your classes)
diseases_tomato = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Function to load and transform an image
def load_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return transform(image).unsqueeze(0).to(device)
    except UnidentifiedImageError:
        return None

# Function to predict the class of an image
def predict_image(img_tensor):
    with torch.no_grad():
        preds = model(img_tensor)
    _, predicted = torch.max(preds, 1)
    return diseases_tomato[predicted.item()]

# Define the main route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scan-tomat')
def scan():
    return render_template('scan.html', image_id=uuid.uuid4())

# Define the predict route with a special URL pattern
@app.route('/predict/<image_id>', methods=['POST'])
def predict(image_id):
    if 'file' not in request.files:
        return "Please upload a file", 400
    file = request.files['file']
    if file and file.filename != '':
        # Save the uploaded file to static/user_uploaded
        filename = f"{image_id}_{file.filename}"
        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)

        # Reopen the saved file for reading
        with open(file_path, 'rb') as f:
            img_bytes = f.read()
        
        img_tensor = load_image(img_bytes)
        if img_tensor is None:
            return "The uploaded file is not a valid image", 400

        prediction = predict_image(img_tensor)

        # Define the mapping from prediction to template
        template_mapping = {
            "Tomato___Bacterial_spot": "bacterial_spot.html",
            "Tomato___Early_blight": "early_blight.html",
            "Tomato___Late_blight": "late_blight.html",
            "Tomato___Leaf_Mold": "leaf_mold.html",
            "Tomato___Septoria_leaf_spot": "septoria_leaf_spot.html",
            "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites.html",
            "Tomato___Target_Spot": "target_spot.html",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus.html",
            "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus.html",
            "Tomato___healthy": "healthy.html"
        }
        output_page = template_mapping.get(prediction, "error.html")
        return render_template(output_page, pred_output=prediction, user_image=filename)
    return "No file uploaded", 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
