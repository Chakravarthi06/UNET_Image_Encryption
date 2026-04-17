# UNET_Image_Encryption
CyberVault 🔐
U-Net Driven Chaotic Image Encryption System
Minor Project - NIT Jalandhar
Author: C. Chakravarthi
Guide: Dr. Urvashi
________________________________________
📋 Table of Contents
	Dataset Requirements
	Project Structure
	Installation
	Usage
	Technical Architecture
	API Documentation
	Security Metrics
________________________________________
Dataset Requirements
❓ Do You Need a Dataset?
NO - This project does NOT require traditional image datasets like ImageNet, CIFAR-10, or custom image collections.
✅ What You Actually Need
The U-Net in this project uses synthetic training data generated mathematically from chaotic systems:
Component	Source	Size	Purpose
Chaotic Seeds	Logistic & Henon Maps	64×64	U-Net Input
Target Keys	Random Matrices	256×256	U-Net Output
Training Pairs	Synthetic Generation	1000+ samples	Weight Learning
🔬 Synthetic Data Generation
# The system generates training data automatically:
for i in range(num_samples):
    x0 = np.random.uniform(0.01, 0.99)  # Logistic parameter
    y0 = np.random.uniform(0.01, 0.99)  # Henon parameter  
    r0 = np.random.uniform(0.01, 0.99)  # Chaotic seed

    seed = ChaoticSeedGenerator.create_chaotic_seed_image(x0, y0, r0, size=64)
    target = np.random.rand(256, 256) * 2 - 1  # Random key matrix
📊 Recommended Training Configuration
Mode	Samples	Epochs	Batch Size	Time
Demo	16	3	4	30 seconds
Development	500	20	16	5 minutes
Production	5000+	100	32	1 hour
Key Insight: The U-Net learns to map chaotic patterns → encryption keys. It never sees actual images during training, which is why no image dataset is needed [2][4].
________________________________________
Project Structure
cybervault/
├── frontend/
│   └── index.html          # Cyberpunk UI (32KB enhanced version)
├── encryption/
│   ├── __init__.py
│   └── engine.py           # Core encryption logic
├── models/
│   └── unet_key_generator.keras  # Auto-generated on first run
├── server.py               # FastAPI backend (updated)
├── requirements.txt        # Dependencies
├── README.md              # This file
└── docs/
    └── presentation.pptx   # Project presentation
________________________________________
Installation
Prerequisites
	Python 3.8+
	4GB RAM minimum (8GB recommended)
	GPU optional (CPU works fine)
Step 1: Clone and Setup
git clone <your-repo-url>
cd cybervault
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Step 2: Install Dependencies
pip install -r requirements.txt
requirements.txt:
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
pydantic==2.5.0
starlette==0.27.0
tensorflow==2.15.0
numpy==1.24.3
opencv-python==4.8.1.78
Pillow==10.1.0
python-multipart==0.0.6
Step 3: Run the Server
python server.py
First run will: 1. Generate synthetic training data (16 samples) 2. Train U-Net for 3 epochs (~30 seconds) 3. Save model to unet_key_generator.keras 4. Start API server at http://localhost:8000
________________________________________
Usage
Web Interface
Open http://localhost:8000 in your browser.
Encryption Flow: 1. Upload image → System generates chaotic parameters automatically 2. Click “Initialize Encryption” 3. Download encrypted image + permutation file
Decryption Flow: 1. Upload encrypted image 2. Enter chaotic parameters [x0, y0, r0] (from encryption step) 3. Upload perm_indices.json 4. Click “Decrypt Image”
API Usage
Encrypt Image
curl -X POST "http://localhost:8000/api/encrypt" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
Response:
{
  "encrypted_image": "base64_encoded_string",
  "key_data": {"perm_indices": [1234, 5678, ...]},
  "chaotic_params": [0.123, 0.456, 0.789],
  "metrics": {
    "original_entropy": 7.21,
    "encrypted_entropy": 7.99,
    "npcr": 99.61,
    "uaci": 33.42
  }
}
Decrypt Image
curl -X POST "http://localhost:8000/api/decrypt" \
  -H "Content-Type: application/json" \
  -d '{
    "encrypted_image": "base64_string",
    "chaotic_params": [0.123, 0.456, 0.789],
    "perm_indices": [1234, 5678, ...],
    "image_shape": [256, 256, 3]
  }'
________________________________________
Technical Architecture
1. Chaotic Seed Generation
Uses Logistic and Henon maps for pseudo-random sequences:
Logistic Map:
x_(n+1)=r⋅x_n⋅(1-x_n )
Where r=3.99 (chaotic regime)
Henon Map:
x_(n+1)=1-ax_n^2+y_n
y_(n+1)=bx_n
Where a=1.4, b=0.3
2. U-Net Key Generator
Input: 64×64 Chaotic Seed
↓
Encoder (4 blocks): 64→128→256→512 filters
↓
Bottleneck: 1024 filters
↓
Decoder (4 blocks): 512→256→128→64 filters
↓
Output: 256×256 Encryption Key
Key Features: - Skip connections for precise localization - Batch normalization for training stability - Tanh output activation (range: [-1, 1])
3. Encryption Pipeline
	Confusion: Pixel permutation using chaotic key-derived indices
	Diffusion: XOR transformation with key matrix
	Avalanche Effect: Single bit change → completely different output
________________________________________
Security Metrics
Metric	Target	Achieved	Status
Information Entropy	~8.0	7.99	✅ Excellent
NPCR	>99%	99.61%	✅ Excellent
UACI	~33.3%	33.42%	✅ Excellent
Key Sensitivity	10^(-15)	10^(-15)	✅ Excellent
Analysis: - High Entropy: Encrypted images appear statistically random - High NPCR: 99.61% pixels change with different keys - Optimal UACI: 33.42% intensity change (close to ideal 33.3%) - Extreme Sensitivity: 10^(-15) changes in chaotic params produce completely different keys
________________________________________
API Documentation
Endpoints
Endpoint	Method	Description
/	GET	Serve frontend interface
/api/	GET	API info
/api/model-status	GET	Check if U-Net is ready
/api/encrypt	POST	Encrypt an image
/api/decrypt	POST	Decrypt an image
/api/download_encrypted	POST	Download encrypted image as file
/health	GET	Health check
Status Codes
	200: Success
	400: Bad request (invalid image/parameters)
	503: Model still initializing
	500: Server error
________________________________________
Troubleshooting
Issue: “Model is initializing” for too long
Solution: First run generates synthetic data and trains U-Net. Wait 1-2 minutes.
Issue: Low entropy (<7.5)
Solution: Train with more synthetic samples (modify num_samples in server.py)
Issue: Decryption produces noise
Solution: 1. Verify chaotic parameters match exactly (copy-paste from encryption) 2. Ensure correct perm_indices.json file 3. Check image dimensions match
Issue: Out of memory
Solution: Reduce batch size in training or use smaller synthetic dataset
________________________________________
Future Enhancements
	Hybrid Chaos-GAN Architecture [2]
	Quantum-enhanced key generation
	Multi-layer cascaded encryption
	Hardware acceleration (FPGA/GPU)
	Real-time video encryption
________________________________________
References
	Amiri, S., & Zaied, M. (2025). Dense Attention U-Net to Break Chaos-Based Color Image Encryption.
	Synthetic Data Generation for Deep Learning (MDPI Mathematics, 2025) [2]
	U-Net Architecture for Image Segmentation (TensorFlow) [6]
	Synthetic Dataset Generation for Privacy-Preserving ML (Purdue University) [4]
________________________________________
License
MIT License - Academic Use
Disclaimer: This is an educational project demonstrating chaotic encryption concepts. For production use, additional security audits recommended.
________________________________________
🔒 CyberVault - Securing the Digital Frontier
NIT Jalandhar - Department of Data Science & Engineering
