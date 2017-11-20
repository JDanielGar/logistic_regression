def ProcessImage(image_dir):
    img = Image.open(str(image_dir)).convert('L')
    img = img.resize((48, 48))
    return np.array(list(img.getdata()))