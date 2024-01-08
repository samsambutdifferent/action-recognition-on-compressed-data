
import subprocess
import os
from os import listdir

INPUT_FORMAT = 'webm'
OUTPUT_FORMAT = 'avi'

if __name__=="__main__":
    target_dir = "./data/" + OUTPUT_FORMAT + "/"
    print(os.listdir())
    cat_dir = "./data/20bn-something-something-v2/"

    failed = []
    for f in listdir(cat_dir):
        f2 = f.replace("." + INPUT_FORMAT, "." + OUTPUT_FORMAT)
        try: 
            if not os.path.exists(target_dir + "/" + f2):
                subprocess.call(
                    f'ffmpeg -i {cat_dir}/{f} {target_dir}/{f2}',
                    shell=True
                )
            else:
                print(f"{f2} already exists")
        except:
            failed.append(f"{f}")
    
    print(f"failed: {str(failed)}")
