from setuptools import find_packages, setup

setup(
    name="CITNP",
    version="0.0.1",
    packages=[
        "CITNP",
        *find_packages(
            include=["causal_active_learning", "causal_active_learning.*"]
        ),
    ],
    package_data={
        "causal_active_learning.integrations.citnp": ["configs/*.yaml"],
    },
)
