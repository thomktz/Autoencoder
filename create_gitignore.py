### The dataset is stored in the Git file, so all 70.000 images appear in source control
# %%
PATH = "images\\ImageFolder\\"
print(PATH)
gitignore = open(".gitignore2", "w+")

for i in range(70000):
    gitignore.write(PATH + str(i).zfill(5) + ".png\n")


PATH2 = "subset\\ImageFolder\\"


for i in range(1000):
    gitignore.write(PATH + str(i).zfill(5) + ".png\n")

gitignore.close()
# %%
