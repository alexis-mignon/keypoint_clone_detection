from setuptools import setup

setup(
    name='keypoint_clone_detection',
    version="0.0.1",
    py_modules=['keypoint_clone_detection'],
    python_requires=">=3.6",
    install_requires=[
        "numpy", "scipy", "scikit-image"
    ],
)
