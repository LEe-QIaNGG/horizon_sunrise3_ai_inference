import torch.nn as nn
import models

class Net(nn.Module):
    def __init__(self,attention_dim, embed_dim, decoder_dim, vocab_size,encoder,decoder):
        super(Net, self).__init__()
        self.encoder=models.Encoder()
        self.encoder.load_state_dict(encoder.state_dict())
        self.decoder=models.DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size)
        self.decoder.load_state_dict(decoder.state_dict())


    def forward(self, image,encoded_captions, caption_lengths,predictions,alphas):
        x=self.encoder(image)
        x=self.decoder(x,encoded_captions, caption_lengths,predictions,alphas)
        return x