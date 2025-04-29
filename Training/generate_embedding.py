# Yous should change for the dataset you want to train on 
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
from tqdm import tqdm
import argparse
def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct, 
                    max_residues=4000, max_seq_len=1000, max_batch=100 ):

    if sec_struct:
        sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(), 
                "protein_embs" : dict(),
                "sec_structs" : dict() 
                }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1)):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)

            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: # in case you want to predict secondary structure from embeddings
                d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # get classification results
                    results["sec_structs"][identifier] = torch.max( d3_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results
 #OPTIONAL 1 REWRITE THIS WITH TFRECORD      IF EMBEDDING FINSIHED DONT CHANGE
def save_embeddings(emb_dict,out_path,write_option = "w"):
    with h5py.File(str(out_path), write_option) as hf:
        for sequence_id, embedding in emb_dict.items(): #
            print(sequence_id)
            print(type(embedding))
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None #OPTIONAL 1 REWRITE THIS WITH TFRECORD      IF EMBEDDING FINSIHED DONT CHANGE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "fasta_file", type=str,
        help = "fasta file"
    )
    parser.add_argument(
      "embedding_residue",type = str,
       help = "embedding residue string" 
     )
    parser.add_argument(
      "embedding_protein",type = str,
       help = "embedding protein string" 
     )
    parser.add_argument(
    "--batch_size",type = int, default=128,
    help = "batch size for loop" 
    )
  
    args = parser.parse_args()       
    seq_path =  args.fasta_file

    per_residue = True 
    per_residue_path = args.embedding_residue# where to store the embeddings

    # whether to retrieve per-protein embeddings 
    # --> only one 1024-d vector per protein, irrespective of its length
    per_protein = True
    per_protein_path = args.embedding_protein # where to store the embeddings

    # whether to retrieve secondary structure predictions
    # This can be replaced by your method after being trained on ProtT5 embeddings
    sec_struct = False
    sec_struct_path = "ss3_preds.fasta" # file for storing predictions

    # make sure that either per-residue or per-protein embeddings are stored
    assert per_protein is True or per_residue is True or sec_struct is True, print(
        "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")

   


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))

  

   
   
#OPTION 2 HDF5 stackoverflow get specific key instead of loading each of them

    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    model, tokenizer = get_T5_model()
    ''' 
    seqs = dict()
    for seq_record in SeqIO.parse(seq_path, "fasta"):
    if len(seq_record.seq) > 1200:
        seqs[seq_record.id] = str(seq_record.seq[:1200])
    else:
        seqs[seq_record.id] = str(seq_record.seq)
    '''
# Load example fasta.
    seqs = read_fasta( seq_path )
    for id, seq in seqs.items():
        if len(seq) > 1200:
            seqs[id] = seq[:1200]

    #OPTION 2 MAYBE DELETE THIS AND ADD OPT2 RESULTS = ENTIRE EMBEDDINGS
    # Compute embeddings and/or secondary structure prediction
    #DELETE RESULTS MAYBE HAS ALL EMBEDDINGS LOADED AGAIN LATER
    seq_ids = list(seqs.keys())
    for i in range(0,len(seqs),args.batch_size):
        subseq_ids = seq_ids[i:i+args.batch_size]
        subseqs = {id: seqs[id] for id in subseq_ids}
        results = get_embeddings( model, tokenizer, subseqs,
                            per_residue, per_protein, sec_struct)
    # Store per-residue embeddings
        if i == 0:
            write_option = "w"
        else:
            write_option = "a"
        if per_residue:
            save_embeddings(results["residue_embs"], per_residue_path,write_option)
        if per_protein:
            save_embeddings(results["protein_embs"], per_protein_path,write_option)
        del results
        


    


