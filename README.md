# Azure Kinect Image Manipulation

A simplified Python toolkit based on [**pyKinectAzure**](https://github.com/ibaiGorordo/pyKinectAzure) for easy manipulation of Azure Kinect infrared (IR) and depth images, tailored for noise reduction and improved skeletal tracking.

## Overview

This repository offers image pre-processing tools for Azure Kinect data, focusing on reducing passive IR noise caused by reflective markers in concurrent validation studies.

## Features

- IR and depth image manipulation tools
- Passive noise reduction techniques (e.g., inpainting)
- Simple example scripts for getting started quickly

## Prerequisites

- [Azure Kinect Sensor SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)

## Installation
```bash
pip install -r requirements.txt 
```

See original pyKinectAzure repository for more details ([pyKinectAzure](https://github.com/ibaiGorordo/pyKinectAzure)).


## Usage
git clone <your-repo-link>
cd <your-repo-name>/examples
python examplePlayBackBodyTrackerDenoiseIR.py

