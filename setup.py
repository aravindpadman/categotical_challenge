import subprocess

# subprocess.call("./gitconfig.sh", shell=True)
# print("git config setup")
# try:
#     subprocess.call("./install_rapids.sh", shell=True)
#     print("rapids installed")
# except:
#     print("Gpu not available:rapids installation failed")
# 
with open("./gitconfig.sh", 'r') as file:
    line = file.readline()
    subprocess.run(line.split())

with open("./install_rapids.sh", 'r') as file:
    try:
        line = file.readline()
        print(line)
        subprocess.run(line.split())
    except Exception as error:
        raise(error)


import sys
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

if __name__ == "__main__":
    try:
        import cuml, cupy, cudf
    except ImportError:
        print("Rapids installation failed")