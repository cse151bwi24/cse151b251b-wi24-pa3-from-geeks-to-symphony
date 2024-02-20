from util import *
from constants import *
from train import *
from dataLoader import *
import torch.optim as optim
import torch.nn as nn
import sys
import os
import util


def train(model, data, data_val, char_idx_map, config, device):
	print('device', device)
	N_EPOCHS = config["no_epochs"]
	LR = config["learning_rate"]
	SAVE_EVERY = config["save_epoch"]
	MODEL_TYPE = config["model_type"]
	HIDDEN_SIZE = config["hidden_size"]
	DROPOUT_P = config["dropout"]
	SEQ_SIZE = config["sequence_size"]
	CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(MODEL_TYPE, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)

	model = model.to(device) # TODO: Move model to the specified device

	optimizer = optim.SGD(model.parameters(), lr=LR) # TODO: Initialize optimizer

	criterion = nn.CrossEntropyLoss() # TODO: Initialize loss function
	# Lists to store training and validation losses over the epochs
	train_losses, validation_losses = [], []

	# Training over epochs
	for epoch in range(N_EPOCHS):
	    # TRAIN: Train model over training data
		totalTrainLossPerEpoch = 0
		totalValLossPerEpoch = 0
		for i in range(len(data)):
			'''
	    	TODO: 
	    		- For each song:
	    			- Zero out/Re-initialise the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.) (Done for you)
	    			- Zero out the gradient (Done for you)
	    			- Get a random sequence of length: SEQ_SIZE from each song (check util.py)
	    			- Iterate over sequence characters : 
	    				- Transfer the input and the corresponding ground truth to the same device as the model's
	    				- Do a forward pass through the model
	    				- Calculate loss per character of sequence
	    			- backpropagate the loss after iterating over the sequence of characters
	    			- update the weights after iterating over the sequence of characters
	    			- Calculate avg loss for the sequence
	    		- Calculate avg loss for the training dataset and 


	    	'''
			model.init_hidden() # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)
			model.zero_grad()   # Zero out the gradient

	        #TODO: Finish next steps here
			input_mask, output_mask = util.get_random_song_sequence_target(data[i], char_idx_map, SEQ_SIZE)
# 			print('input_mask', input_mask)
# 			print('output_mask', output_mask)
# 			print('output_mask shape', output_mask.shape)
			totalLoss = 0
			for index in range(SEQ_SIZE-1):
				input_item = input_mask[index].to(device)
				output_item = output_mask[index].to(device)
				pred, _ = model(input_item)
				pred = pred.to(device)
# 				print("input",input_item)
# # # 				print("output_item shape",output_item.shape)
# 				print("output_item",output_item)
				loss = criterion(pred, output_item)
				totalLoss += loss
			totalLoss.backward()   
			optimizer.step()
			optimizer.zero_grad()
			avg_loss_per_sequence = totalLoss / (SEQ_SIZE-1)
			totalTrainLossPerEpoch += avg_loss_per_sequence


	    	# Display progress
			msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/len(data)*100, i, avg_loss_per_sequence)
			sys.stdout.write(msg)
			sys.stdout.flush()
		print()
		train_losses.append((totalTrainLossPerEpoch/len(data)).item())



	    # TODO: Append the avg loss on the training dataset to train_losses list
        
	    # VAL: Evaluate Model on Validation dataset
		model.eval() # Put in eval mode (disables batchnorm/dropout) !
		with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
	    	# Iterate over validation data
			for i in range(len(data_val)):
				'''
		    	TODO: 
		    		- For each song:
		    			- Zero out/Re-initialise the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.) (Done for you)
		    			- Get a random sequence of length: SEQ_SIZE from each song- Get a random sequence of length: SEQ_SIZE from each song (check util.py)
		    			- Iterate over sequence characters : 
		    				- Transfer the input and the corresponding ground truth to the same device as the model's
		    				- Do a forward pass through the model
		    				- Calculate loss per character of sequence
		    			- Calculate avg loss for the sequence
		    		- Calculate avg loss for the validation dataset and 
		    	'''
				model.init_hidden() # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)

	    		#TODO: Finish next steps here
				input_mask_val, output_mask_val = util.get_random_song_sequence_target(data_val[i], char_idx_map, SEQ_SIZE)
				totalLoss = 0
				for index in range(SEQ_SIZE-1):
					input_item_val = input_mask_val[index].to(device)
					ouput_item_val = output_mask_val[index].to(device)
					pred_val, _ = model(input_item_val)
					pred_val = pred_val.to(device)
					loss_val = criterion(pred_val, ouput_item_val)
					totalLoss += loss_val
				avg_loss_per_sequence_val = totalLoss/(SEQ_SIZE-1)
				totalValLossPerEpoch += avg_loss_per_sequence_val


		    	# Display progress
				msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/len(data_val)*100, i, avg_loss_per_sequence_val)

				sys.stdout.write(msg)
				sys.stdout.flush()
			print()

		# TODO: Append the avg loss on the validation dataset to validation_losses list
		validation_losses.append((totalValLossPerEpoch/len(data_val)).item())

		model.train() #TURNING THE TRAIN MODE BACK ON !
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')

	    # Save checkpoint.
		if (epoch % SAVE_EVERY == 0 and epoch != 0)  or epoch == N_EPOCHS - 1:
			print('=======>Saving..')
			torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, './checkpoint/' + CHECKPOINT + '.t%s' % epoch)

	return train_losses, validation_losses

