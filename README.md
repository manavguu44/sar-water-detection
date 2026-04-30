## Learning SAR

Synthetic Aperture Radar, or SAR, is fundamentally different from the optical imagery we see on platforms like Google Maps. While optical satellites act like high-powered cameras that rely on sunlight to capture a "picture," SAR is an active system. It carries its own energy source, beaming microwave pulses down to the surface and measuring the "echo" that bounces back. Because it doesn't rely on light, SAR can "see" in total darkness. More importantly, because microwaves can pass through the atmosphere regardless of weather, SAR can see through thick clouds, smoke, and rain—conditions that leave optical sensors completely blind. This makes it an indispensable tool for emergency response, where waiting for a clear, sunny day isn't an option.

When we use SAR data for Machine Learning, we have to look past the "pixels" and consider the physics of the radar bounce. One of the most important concepts is polarization, usually labeled as VV or VH. This tells us the orientation of the radar wave. A VV signal is great for seeing the roughness of the ground or water, while a VH signal is essential for identifying complex structures like forests. In a forest, the radar waves bounce off many branches and leaves, "twisting" the signal's orientation before it returns to the satellite. This is why a U-Net model can be so effective; by looking at both channels, it can distinguish between a flat, smooth road (which looks dark) and a dense, messy forest (which looks bright).

However, SAR data isn't perfect "out of the box." It comes with a unique type of noise called speckle, which gives the images a grainy, salt-and-pepper appearance. This isn't a sensor error, but rather a result of interference between the radar waves as they bounce off tiny objects near each other. For a computer vision model, this graininess can be a major distraction, often requiring a preprocessing step called "despeckling" to smooth the image while preserving the sharp edges of objects. Additionally, we have to account for the incidence angle—the angle at which the radar hits the ground. If the angle is too steep or shallow, it can create shadows behind mountains or distort the shape of buildings, which can confuse a model that hasn't seen those specific geometries before.

The real-world applications for this technology are incredibly impactful, particularly in flood mapping. Because smooth water acts like a mirror, it reflects the radar signal away from the satellite, making flooded areas appear as distinct black shapes in the data. Even in the middle of a hurricane, SAR can provide a clear map of where the water is. Beyond disasters, SAR is used for land-cover mapping to track deforestation in cloudy tropical regions and for monitoring "subsidence"—using the precision of radar to detect if a city or a bridge is sinking by just a few millimeters. By combining the structural intelligence of a U-Net with a deep understanding of these radar physics, we can build tools that monitor our changing planet more reliably than ever before.

Synthetic Aperture Radar (SAR) data has been widely used with machine learning techniques for applications such as land-cover classification, water detection, flood mapping, and urban analysis. In this project, I explored both classical machine learning approaches and modern deep learning methods applied to SAR data in order to understand their strengths, limitations, and suitability for different tasks.

## Exploring SAR ML models and picking a use case

Classical machine learning methods rely on handcrafted features derived from SAR imagery. These include backscatter intensity, texture features such as those derived from gray-level co-occurrence matrices (GLCM), and polarization information such as VV and VH channels. These features are then used with models such as Random Forests and Support Vector Machines. Random Forest is widely used for SAR-based classification tasks due to its simplicity, robustness, and ability to perform well with limited data. However, it operates at the pixel level and does not capture spatial context. Support Vector Machines are also effective for classification tasks and perform well when the feature space is well defined, but they require careful feature engineering and do not scale easily to large datasets.

Deep learning approaches, particularly convolutional neural networks, have significantly improved performance in SAR-based tasks by learning features directly from raw data. Convolutional Neural Networks (CNNs) are used for classification and feature extraction, while segmentation models such as UNet are widely used for pixel-wise prediction tasks. UNet is particularly effective for SAR segmentation because it captures both local and global spatial patterns, allowing it to identify structures such as rivers, lakes, and flooded regions. However, these models require labeled data and are computationally more intensive than classical approaches.

In addition to task-specific models, there has been increasing interest in foundation and pre-trained models for geospatial data. Models such as SatMAE and Prithvi are trained on large-scale satellite imagery and can be fine-tuned for downstream tasks. These models aim to learn general representations of Earth observation data, including SAR, and can improve performance when labeled data is limited. Benchmark datasets such as SEN12MS, which combines SAR and optical imagery, are commonly used for evaluating such models on tasks like land-cover classification.

For this project, I selected water detection using SAR data as the primary use case. This task is well-suited to SAR because water surfaces typically exhibit low backscatter and appear dark in SAR imagery, making them relatively easy to distinguish from other land-cover types. Additionally, SAR can operate in cloudy and low-light conditions, making it particularly useful for flood monitoring and disaster response. The problem is formulated as a binary segmentation task, where the input is a Sentinel-1 SAR image and the output is a pixel-wise classification of water versus non-water.

To address this task, I implemented multiple approaches. First, I developed a Random Forest model as a baseline, using SAR backscatter values for pixel-wise classification. Next, I implemented a UNet-based deep learning model to perform spatial segmentation, allowing the model to capture contextual information beyond individual pixels. I then extended this approach by incorporating both VV and VH polarization channels as input, enabling the model to leverage additional information about surface scattering properties. Finally, I implemented a temporal UNet model that uses multi-date SAR data as input, allowing the model to incorporate temporal variation in backscatter values.

An important observation from this exploration is that the performance improvements from more advanced models were limited by the quality of the training labels. Since the water masks were generated using simple thresholding on SAR backscatter, they represent weak supervision rather than true ground truth. As a result, even complex models tend to learn patterns similar to the underlying threshold rule. Furthermore, evaluation on a different geographic region (Mumbai) revealed a drop in performance, highlighting the challenge of generalization across regions with different SAR characteristics.

This survey and implementation highlight the trade-offs between classical and deep learning approaches for SAR data, as well as the importance of high-quality labels and diverse training data for achieving robust performance.

# Data Collection and Preprocessing

The data used in this project was obtained from the Sentinel-1 satellite mission, which provides Synthetic Aperture Radar (SAR) imagery. The data was accessed through Google Earth Engine, which allows efficient querying, filtering, and export of satellite datasets. Sentinel-1 provides dual-polarization SAR data (VV and VH), which was used as the primary input for all models in this project.

The area of interest included two regions: Delhi, which was used for model development, and Mumbai, which was used for evaluating generalization. For Delhi, a rectangular region covering the urban and surrounding areas was selected, and SAR data from the year 2020 was used. For Mumbai, a similar approach was followed, selecting a coastal urban region for testing. In addition to single-date imagery, temporal data was also extracted for multiple months (January to April 2020) to support temporal modeling.

Basic preprocessing steps were applied to prepare the data for machine learning. The SAR backscatter values were first handled for missing or invalid values by replacing NaNs with a default low value. The data was then normalized from its original decibel range (approximately -25 to 0) into a 0 to 1 range to make it suitable for model training. For deep learning models, the full SAR images were divided into smaller patches of fixed size (e.g., 128×128) to enable efficient training and to allow the model to learn localized spatial patterns. For temporal modeling, multiple SAR images across different dates were stacked along the channel dimension to form multi-channel inputs. Ground truth labels were generated using a simple threshold on SAR backscatter values, producing binary masks indicating water and non-water regions.

These preprocessing steps ensured that the data was consistent, normalized, and structured appropriately for both classical machine learning and deep learning pipelines.

The app screenshot - ( for mumbai aoi) 
![alt text](image.png)

for Delhi - 

![alt text](image-1.png)

## Basic Instructions

1. Set up the environment

- Clone the repository and open the project folder in VS Code
- Create a virtual environment:

    python -m venv .venv

    Activate the environment (Windows):

    .\.venv\Scripts\Activate.ps1

    Install required packages:

    python -m pip install rasterio numpy matplotlib scikit-learn joblib torch torchvision tqdm opencv-python fastapi uvicorn pillow

2. Prepare the data

    Place the following files inside the data/ folder:

    sar_vv_vh.tif
    water_mask.tif
    delhi_temporal_vv_vh.tif
    mumbai_vv_vh.tif
    mumbai_mask.tif
    mumbai_temporal_vv_vh.tif

3. Run training

    Train Random Forest:

    python -u src/train_random_forest.py

    Train UNet:

    python -u src/create_patches.py
    python -u src/train_unet.py

    Run UNet inference:

    python -u src/infer_unet_full.py

    rain Temporal UNet:

    python -u src/create_temporal_patches.py
    python -u src/train_temporal_unet.py
    python -u src/infer_temporal_unet_full.py

4. Run Mumbai testing

    python -u src/test_rf_mumbai.py
    python -u src/test_unet_mumbai.py
    python -u src/test_temporal_unet_mumbai.py
    python -u src/create_mumbai_comparison.py

5. Generate final outputs
    python -u src/create_final_comparison.py
    python -u src/create_map_overlays.py

6. Start backend
    python -m uvicorn backend.main:app --reload
    Backend will run at:

    http://127.0.0.1:8000

7. Start frontend
    Open in browser:

    frontend/index.html

8. Trigger sample inference

    Select Delhi or Mumbai

    Click Run Inference

    Click Final Comparison

    Click Show Metrics

    Click Show Map Overlay


The current implementation uses simple script-based configuration. Paths and parameters can be easily adapted or extended using CLI arguments or config files