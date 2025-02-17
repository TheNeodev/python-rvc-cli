from setuptools import setup, find_packages
import sys

# Base dependencies
install_requires = [
    "requests>=2.31.0,<2.32.0",
    "tqdm",
    "wget",
    "ffmpeg-python>=0.2.0",
    "faiss-cpu==1.7.3",
    "soundfile==0.12.1",
    "noisereduce",
    "pedalboard",
    "stftpitchshift",
    "yt-dlp==2025.01.26",
    "audio-separator[gpu]==0.28.5",
    "torch==2.3.1",
    "torchaudio==2.3.1",
    "torchvision==0.18.1",
    "torchcrepe==0.0.23",
    "torchfcpe",
    "libf0",
    "transformers==4.44.2",
    "matplotlib==3.7.2",
    "tensorboard",
    "tensorboardX",
    "pypresence",
    "beautifulsoup4",
    "flask",
]

# Platform-specific dependencies
if sys.platform == "darwin":  # macOS
    install_requires.extend([
        "pip>=23.3",
        "wheel",
        "PyYAML",
        "omegaconf>=2.0.6",
        "certifi>=2023.07.22",
        "antlr4-python3-runtime==4.8",
        "numba==0.57.0",  # Needed for librosa
    ])
elif sys.platform == "linux":
    install_requires.append("numba>=0.57.0")
elif sys.platform == "win32":
    install_requires.append("numba==0.57.0")

# Optional dependencies
extras_require = {
    "dev": ["pytest", "black", "flake8"],
    "docs": ["sphinx", "sphinx_rtd_theme"],
}

# Setup configuration
setup(
    name="rvc_cli",
    version="1.5.0",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    readme="README.md",
    license="MIT",
    packages=find_packages(include=["rvc_cli", "rvc_cli.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "rvc-cli = rvc_cli.core:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
