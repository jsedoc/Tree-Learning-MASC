
# coding: utf-8

# In[ ]:


#get_ipython().system('pip3 install tqdm')
#get_ipython().run_line_magic('run', "'tree2seq_dataloader.ipynb'")
#get_ipython().run_line_magic('run', "'childsumtree.ipynb'")
#get_ipython().run_line_magic('run', "'decoderrnn.ipynb'")

from tree2seq import *
#from tree2seq_dataloader import *
#from childsumtree import *
#from decoderrnn import *
#from util import *
# In[ ]:


from torch import optim


def train(input_variable, target, 
          encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = len(target)

    loss = 0
    predicted = []
    
    if 1: #tree_encoder:
        encoder_hidden = encoder(input_variable[0], Variable(input_variable[1]))
    else:
        # TBD ...
        encoder_hidden = encoder.initHidden()
        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        input_length = input_variable.size()[0]
        

    decoder_input = Variable(torch.LongTensor([[target[0]]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden[1].resize(1, 
                                              encoder_hidden[1].size()[0], 
                                              encoder_hidden[1].size()[1])
    
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        target_variable = Variable(torch.LongTensor([[target[di]]]))
        loss += criterion(decoder_output, target_variable[0][0])
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        predicted.append(ni)
        decoder_input = target_variable  # Teacher forcing
    # call backward

    loss.backward()

    # update parameters
  
    encoder_optimizer.step()
    decoder_optimizer.step()

    return (loss.data[0] / target_length, predicted)


# In[ ]:


dataset = Dataset()
dataset.create_vocab('data/train.orig')
one_hot_dict = dataset.create_one_hot()
    #print(one_hot_dict['('])

trees = dataset.read_trees('data/train.orig')
seqs = dataset.read_seqs('data/train.orig')

ptr_trees = [dataset.make_ptr_tree(tree, dataset.vector_dim) for tree in tqdm(trees)]
vec_seqs = []
for seq in seqs:
    vec_seqs.append([dataset.token_dict[x] for x in seq.split()])


input_size = len(dataset.vocab.keys()) # here also equal to length of output vocab
hidden_size = 128 # any size can be given here
output_size = input_size

encoder = ChildSumTreeLSTM(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size)

learning_rate = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()


print_every = 100
plot_every = 100
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every
n_iters = 50

iteration_count = 0

for k in range(n_iters):
    for i in range(len(trees)):
        iteration_count += 1
        
        input_variable = ptr_trees[i]
        target_variable = vec_seqs[i]

        loss, predicted = train(input_variable, target_variable, encoder, decoder, 
                                encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if (i+1) % print_every == 0:
            print('input: ', trees[i])
            print('target: ', seqs[i][:-1])
            print('pred: ', " ".join([dataset.idx2token[x] for x in predicted]))
            # print('pred: ', predicted))
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('iterations: %d, time elapsed: %s, loss:%.4f' % 
                    (iteration_count, timeSince(start, i / (n_iters)),
                                         print_loss_avg))

        if (i+1) % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            print(len(plot_losses))
            plot_loss_total = 0 

