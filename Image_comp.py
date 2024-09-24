import numpy as np
import cv2
import heapq
from copy import deepcopy as cp
from bitarray import bitarray
import time


# O(nlogn + m.h)
class HuffmanNode:
    def __init__(self, value, prob, leaf=False):
        self.value = value
        self.prob = prob
        self.childNodes = [None, None]
        self.leaf = leaf
        self.code = ''

class Image:
    def __init__(self):
        self.inPath = ""
        self.outPath = ""
        
        self.inImage = np.zeros(1)
        self.outImage = np.zeros(1)
        self.image_data = np.zeros(1)
        
        self.row = 0
        self.column = 0
        self.depth = 0
        
        self.histogram = np.zeros(1)
        self.frequency = np.zeros(1)
        
        self.encodedString = ''
        self.prob_dict = {}
        self.allNodes = []
        self.leafNodes = {}
        self.binaryFromFile = []
        self.decodeList = []
        
        self.root = HuffmanNode(-1, -1)
    
    def readImage(self, path):
        # O(n)
        self.inPath = path
        try:
            self.inImage = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            self.row, self.column, self.depth = self.inImage.shape
            
            temp = self.inImage.flatten()
            temp = np.append(temp, self.row)
            temp = np.append(temp, self.column)
            temp = np.append(temp, self.depth)
            
            self.image_data = temp
            
            self.histogram = np.bincount(self.image_data, minlength=max(256, self.row, self.column, self.depth))
            totalCount = np.sum(self.histogram)
            self.frequency = np.array([index for index, count in enumerate(self.histogram) if count != 0])
            
            for pixels in self.frequency:
                self.prob_dict[pixels] = self.histogram[pixels] / totalCount
        except:
            print("Error reading the image!!!")
            
    def writeImage(self, path):
        # O(n)
        self.outPath = path
        try:
            cv2.imwrite(path, self.outImage)
        except:
            print("Error in writing image!!!")
            
    def probability(self, Node):
        return Node.prob
        
    def buildAllNodes(self):
        # O(n)
        for pixel, probability in self.prob_dict.items():
            node = HuffmanNode(pixel, probability, 1)
            self.allNodes.append(node)
            
    def buildTree(self):
        self.buildAllNodes()
        temp = sorted(cp(self.allNodes), key=self.probability)
        
        while True:
            leftNode = temp[0]
            rightNode = temp[1]
            temp.pop(0)
            temp.pop(0)
            
            newNode = HuffmanNode(-1, leftNode.prob + rightNode.prob)
            newNode.childNodes[0] = leftNode
            newNode.childNodes[1] = rightNode
            
            temp = list(heapq.merge(temp, [newNode], key=self.probability))
            
            if newNode.prob == 1.0:
                self.root = newNode
                return
        
    def assignCodes(self, root, code):
        root.code = code
        if root.leaf:
            self.leafNodes[root.value] = root.code
        if root.childNodes[0] is not None:
            self.assignCodes(root.childNodes[0], code + '0')
        if root.childNodes[1] is not None:
            self.assignCodes(root.childNodes[1], code + '1')
            
    def huffmanAlgo(self):
        self.buildTree()
        self.assignCodes(self.root, '')

        flattened_image = self.inImage.flatten()

        encoded_pixels = [self.leafNodes[pixel] for pixel in flattened_image]

        self.encodedString = ''.join([self.leafNodes[self.row], self.leafNodes[self.column], self.leafNodes[self.depth]] + encoded_pixels)       
    
    def decode(self):
        self.binaryFromFile = bitarray(self.encodedString)

        def decode_recursive(root, binary_data, decode_list):
            current_node = root
            for bit in binary_data:
                current_node = current_node.childNodes[bit]
                if current_node.leaf:
                    decode_list.append(current_node.value)
                    current_node = root
                if len(decode_list) == (self.row * self.column * self.depth) + 3:
                    break

        decode_list = []
        decode_recursive(self.root, self.binaryFromFile, decode_list)
        self.decodeList = decode_list

        
    def decodeIm(self):
        self.decode()
        decodeList = self.decodeList
        out_row = decodeList[0]
        decodeList.pop(0)
        out_column = decodeList[0]
        decodeList.pop(0)
        out_depth = decodeList[0]
        decodeList.pop(0)

        out = np.zeros((out_row, out_column, out_depth))

        for i in range(len(decodeList)):
            id = i // out_depth
            x = id // out_column
            y = id % out_column    
            z = i % out_depth
            out[x][y][z] = decodeList[i]
        out = out.astype(dtype=int)
        self.outImage = out
        
    def checkCoding(self):
        return np.all(self.inImage == self.outImage)

    def huffmanCode(self, input_pth, output_pth="./output.png", toCheck=0):
        
        print("Initialising..................")
        self.readImage(input_pth)
        print("Initialized")
        
        start = time.time()
        print('Coding Image started\n')
        self.huffmanAlgo()
        print('Coding Image completed\n')
        end = time.time()
        
        print("\nOriginal Size of Image : ", self.row * self.column * self.depth * 8, " bits")
        print("\nCompressed Size : ", len(self.encodedString), " bits")
        print("\nCompressed factor : ", self.row * self.column * self.depth * 8 / (len(self.encodedString)), "\n")

        print("Took ", end - start, " sec to encode input image\n")
        print('Sending coded data\n')
        print('Coded data sent\n')

        s = time.time()

        print('Started decoding compressed image\n')
        self.decodeIm()
        self.writeImage(output_pth)
        print('Completed decoding compressed image (open output image from the above mentioned path) \n')

        e = time.time()
        print("Took ", e - s, " sec to decode compressed image\n")
        if toCheck:
            print("Are both images same : ", self.checkCoding())

image = Image()
inp_image=input("Enter input image:")
out_image=input("Enter output image:")
image.huffmanCode(inp_image,out_image, toCheck=1)