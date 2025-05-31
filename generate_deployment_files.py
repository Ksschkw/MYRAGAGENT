import os
import pkg_resources
import subprocess
import platform

def generate_requirements_txt():
    """Generate requirements.txt from installed packages in the current environment."""
    installed_packages = [dist.project_name for dist in pkg_resources.working_set]
    with open('requirements.txt', 'w') as f:
        for package in sorted(installed_packages):
            try:
                pkg = pkg_resources.get_distribution(package)
                f.write(f"{package}=={pkg.version}\n")
            except pkg_resources.DistributionNotFound:
                print(f"Warning: Could not find version for {package}")
    print("Generated requirements.txt")

def generate_dockerfile():
    """Generate Dockerfile for Northflank deployment."""
    dockerfile_content = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV GROQ_API_KEY=${GROQ_API_KEY}
CMD ["sh", "start.sh"]
"""
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    print("Generated Dockerfile")

def generate_start_sh():
    """Generate start.sh for running the FastAPI server."""
    start_sh_content = """#!/bin/bash
# Ensure the script is executable on the container
python AgentKosiV2.py --server
"""
    with open('start.sh', 'w') as f:
        f.write(start_sh_content)
    # Only attempt chmod on non-Windows systems
    if platform.system() != 'Windows':
        subprocess.run(['chmod', '+x', 'start.sh'], check=True)
    print("Generated start.sh")

def main():
    """Main function to generate all deployment files."""
    generate_requirements_txt()
    generate_dockerfile()
    generate_start_sh()
    print("All deployment files generated successfully!")

if __name__ == "__main__":
    main()