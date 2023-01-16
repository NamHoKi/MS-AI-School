import os
"""
AFAD_remove
    - ages
        - gender
            -imgs
"""
i = 0
for age in os.listdir('./AFAD_remove/') :
    for gender in os.listdir('./AFAD_remove/' + age + "/") :
        for image in os.listdir('./AFAD_remove/' + age + "/"+gender+"/") :
            os.rename("./AFAD_remove/" + age + "/" + gender+"/"+image,
                    './AFAD_remove/' + age + "/" + gender + "/"+ f"x{i}.png")
            i += 1
