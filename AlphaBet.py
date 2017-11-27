from collections import OrderedDict

class AlphaBet:
    def __init__(self):
        self.list = []
        self.dict = OrderedDict()

    def makeVocab(self, inst):
        # for i in inst:
        #     if i not in self.list:
        #         self.list.append(i)
        # for k in range(len(self.list)):
        #     self.dict[self.list[k]] = k
        # return self.dict
        for k in range(len(inst)):
            self.dict[inst[k]] = k
        return self.dict
    # def maketag(self, inst):
    #     for j in range(len(inst)):
    #         self.tag_dict[inst[j]] = j
    #     return self.tag_dict