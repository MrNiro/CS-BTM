from transformers import BertModel, BertTokenizer
import torch

device = torch.device('cpu')


class BitermBert:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        print("Initializing Biterm Encoder...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name) # .to(device)
        self.bert_model = BertModel.from_pretrained(model_name).to(device)
        print("Init Done!")

    def encode(self, text):
        token = self.tokenizer.encode(text)
        result = self.bert_model(torch.tensor(token).unsqueeze(0))
        return result[1]    # size(1 * 768)

    def cal_biterm_similarity(self, b1: tuple, b2: tuple):
        w1, w2 = b1[0], b1[1]
        w3, w4 = b2[0], b2[1]

        v1 = self.encode(w1)
        v2 = self.encode(w2)
        v3 = self.encode(w3)
        v4 = self.encode(w4)

        similarity = (torch.cosine_similarity(v1, v3) +
                      torch.cosine_similarity(v1, v4) +
                      torch.cosine_similarity(v2, v3) +
                      torch.cosine_similarity(v2, v4)) / 4
        return float(similarity)

    @staticmethod
    def cal_vector_similarity(b1, b2):
        v1, v2 = b1[0], b1[1]
        v3, v4 = b2[0], b2[1]

        similarity = (torch.cosine_similarity(v1, v3) +
                      torch.cosine_similarity(v1, v4) +
                      torch.cosine_similarity(v2, v3) +
                      torch.cosine_similarity(v2, v4)) / 4
        return float(similarity)


if __name__ == '__main__':
    # model_name = 'bert-base-chinese'
    model_name = 'bert-base-multilingual-cased'
    # model_name = 'bert-base-uncased'

    # model_path = './model/multi_cased_L-12_H-768_A-12/'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)      # 这地方好坑

    t1 = tokenizer.encode('希望')
    t2 = tokenizer.encode('想要')
    t3 = tokenizer.encode('程序员')
    t4 = tokenizer.encode('工程师')

    # input_1 = tf.convert_to_tensor(t1)
    input_1 = torch.tensor(t1).unsqueeze(0)
    input_2 = torch.tensor(t2).unsqueeze(0)
    input_3 = torch.tensor(t3).unsqueeze(0)
    input_4 = torch.tensor(t4).unsqueeze(0)

    r1 = bert_model(input_1)
    r2 = bert_model(input_2)
    r3 = bert_model(input_3)
    r4 = bert_model(input_4)

    print(r1[1].size())

    dis_1_2 = torch.cosine_similarity(r1[1], r2[1])
    dis_1_3 = torch.cosine_similarity(r1[1], r3[1])
    dis_3_4 = torch.cosine_similarity(r3[1], r4[1])

    print(dis_1_2)
    print(dis_1_3)
    print(dis_3_4)
