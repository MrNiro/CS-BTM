# -*- coding: utf-8 -*-
from Biterm import *


class Doc:
    """
    @description: 处理文本的类
    @param {type}
    @return:
    """

    def __init__(self, ws):
        self.ws = ws

    def read_doc(self, s):
        for w in s.split(' '):
            self.ws.append(int(w))

    def size(self):
        return len(self.ws)

    def get_w(self, i):
        assert (i < len(self.ws))
        return self.ws[i]

    ''' 
      Extract biterm from a document
        'win': window size for biterm extraction
        'bs': the output biterms
    '''
    def gen_biterms(self, bs, cur_id, win=15):
        if len(self.ws) < 2:
            return
        for i in range(len(self.ws) - 1):
            for j in range(i + 1, min(i + win, len(self.ws))):
                bs.append(Biterm(cur_id, self.ws[i], self.ws[j]))
                cur_id += 1

# if __name__ == "__main__":
#     s = '2 3 4 5'
#     d = Doc(s)
#     bs = []
#     print("test")
#     d.gen_biterms(bs)
#     for biterm in bs:
#         print('wi : ' + str(biterm.get_wi()) + ' wj : ' + str(biterm.get_wj()))
