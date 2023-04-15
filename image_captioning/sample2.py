### RUNNING ONE FOLDER AT A TIME TO PREDICT CAPTIONS

import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import json
import os.path as osp
import csv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)


    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))



    # Open the TRAIN JSON file
    test_caption_path = json.load(open("../AREL-data-process/train_caption_data.json"))
    test_descr={}
    test_base= "../VIST/resnet_features/fc/train"
    for k,value in test_caption_path.items():
        f_path=k+'.npy'
        # Generate an caption from the resnet_features available- that have been used for training purpose also
        image_features = torch.from_numpy(np.load(osp.join(test_base,f_path))).to(device)
        # print(image_features.shape)
        feature = encoder(image_features)
        # feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        t=(value,sentence)
        test_descr[str(k)]=t
        # Print out the image and the generated caption
        # print (sentence)

    # Open the CSV file for writing
    with open('train_description_created.csv', 'w', newline='') as csv_file:
        # Create a CSV writer object
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(['PhotoID', 'Actual Text', 'Predicted Caption'])

        # Write the data rows
        for key, values in test_descr.items():
            writer.writerow([key, values[0], values[1]])

    with open('train_description_created.json', 'w') as f:
        f.write(json.dumps(test_descr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder_best.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder_best.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
