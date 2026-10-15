import os
os.makedirs("./cat_model_save", exist_ok=True)
model.save_model("./cat_model_save/model.cbm")