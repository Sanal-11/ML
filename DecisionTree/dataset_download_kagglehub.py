
def GetDataSet(url):
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download(url)

    print("Path to dataset files:", path)


GetDataSet("ikynahidwin/depression-professional-dataset")

# GetDataSet("gregorut/videogamesales")

# GetDataSet("thedevastator/streaming-activity-dataset")
