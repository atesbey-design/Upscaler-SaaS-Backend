import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F  # F kısayolu burada tanımlandı

# Custom wrapper for RealESRGANer
def get_realesrgan():
    try:
        from realesrgan import RealESRGANer
        return RealESRGANer
    except ImportError:
        # If the import fails, we'll create a dummy class
        class DummyRealESRGANer:
            def __init__(self, *args, **kwargs):
                print("Warning: Using dummy RealESRGANer due to import error.")
            
            def enhance(self, image_path, outscale):
                # Basic upscaling using PyTorch
                img = Image.open(image_path)
                transform = transforms.ToTensor()
                img_tensor = transform(img)

                # Upscaling with PyTorch (bicubic interpolation)
                upscaled_img = F.resize(img_tensor, (int(img_tensor.shape[1] * outscale), int(img_tensor.shape[2] * outscale)), interpolation=F.InterpolationMode.BICUBIC)

                # Convert tensor back to image
                to_pil = transforms.ToPILImage()
                upscaled_img_pil = to_pil(upscaled_img)

                output_path = 'upscaled_image.png'
                upscaled_img_pil.save(output_path)
                return output_path, upscaled_img_pil

        return DummyRealESRGANer

RealESRGANer = get_realesrgan()

app = Flask(__name__)
CORS(app)

model = RealESRGANer(scale=2, model_path='models/RealESRGAN_x2plus.pth') # Adjust model path if needed

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        data = request.get_json()
        image_b64 = data['image']
        image_data = base64.b64decode(image_b64)

        # Load the image into memory using PIL
        image = Image.open(io.BytesIO(image_data))

        # Save the image temporarily (you can handle this in-memory if preferred)
        temp_image_path = 'temp_image.png'
        image.save(temp_image_path)

        # Upscale the image
        output, upscaled_img = model.enhance(image_path=temp_image_path, outscale=2)

        # Encode the upscaled image to base64
        with open(output, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({'upscaled_image': encoded_string})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)
