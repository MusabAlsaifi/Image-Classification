# imports
import os
import json
import torch
import argparse
import matplotlib.pyplot as plt
from utils import imshow
from model_operations import Predict


if __name__ == '__main__':
    
    # load Categories names form JSON file
    cat_to_name = None
    try:
        with open("cat_to_name.json", "r") as cat_file:
            cat_to_name = json.load(cat_file)
    except FileNotFoundError:
        print("Error: The file 'cat_toname.json' was not found.")


    parser = argparse.ArgumentParser(description="Predict the most likely image class and it's associated probability")

    # Requared argument
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("image", type=str, help="The path of the image file")

    # Optional argument 
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="The path of the checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="The top K classes")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if specified")

    args = parser.parse_args()

    # Create the full path of the test dataset
    data_dir = args.data_dir
    test_dir = os.path.join(args.data_dir, "valid")

    # Setting up the device based on the --gpu flag
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    # using GPU for MacOS
    elif args.gpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using {device} device")

    predictor = Predict(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        top_k=args.top_k,
        device=device, 
        test_data=test_dir
    )

    # Use the predict method to get top k predictions
    topk_ps, topk_classes, image = predictor.predict()

    # format the probabilities
    topk_ps = [float(format(prob * 100, ".2f")) for prob in topk_ps.cpu().numpy()]
    
    # if categoies file is not available
    if cat_to_name == None:
        print(f"The category name(s) for {topk_classes} is/are not available")
        print(f"Probabilty of the top k classes(%): {topk_ps}")
    
    # Map the predcited classes to thies category name from cat_to_name.json
    else:
        cat_name = [cat_to_name[cat] for cat in topk_classes]

        # Plotting
        # Create a figure for imshow
        fig1, ax1 = plt.subplots()
        imshow(image[0], title=cat_name[0], ax=ax1)
        fig1.savefig('output_imshow.png')
        plt.show()
        plt.close(fig1)  # close the figure

        # Create another figure for barh
        fig2, ax2 = plt.subplots()
        ax2.barh(cat_name, topk_ps, color='blue')
        ax2.set_xlabel('Probabilities(%)')
        ax2.set_title('Top Predicted Classes')
        fig2.savefig('output_barh.png')
        plt.show()
        plt.close(fig2)  # close the figure
