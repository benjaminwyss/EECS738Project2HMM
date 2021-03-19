import numpy as np
import random

class hmm:

	def __init__(self):
		self.X = None
		pass

	def fit(self, X):
		self.X = X

		# Use a dictionary to map words to states and vice versa
		self.wordStateDict = {"<start>": 0, "<end>": 1}
		self.stateWordDict = ["<start>", "<end>"]
		self.uniqueWords = 2
		
		# Determine all unique words in text corpus X
		for line in self.X:
			for word in line[0].split():
				if word not in self.wordStateDict:
					self.wordStateDict[word] = self.uniqueWords
					self.stateWordDict.append(word)
					self.uniqueWords += 1

		# Initialize state transition matrix
		self.transitionMatrix = np.zeros((self.uniqueWords, self.uniqueWords))
		self.allStates = range(self.uniqueWords)
		
		# Determine probability of state transitions
		for line in self.X:
			prevWord = "<start>"
			for word in line[0].split():
				if word == "<start>":
					continue
				prevState = self.wordStateDict[prevWord]
				currState = self.wordStateDict[word]
				self.transitionMatrix[prevState, currState] += 1
				prevWord = word
		self.transitionMatrix = self.transitionMatrix/self.transitionMatrix.sum(axis=1, keepdims=1)

		
	def generateFakeText(self):
		currState = 0
		fakeText = ""
		while True:
			nextState = np.random.choice(self.allStates, 1, p=self.transitionMatrix[currState])[0]
			
			if nextState == 1:
				break

			fakeText += self.stateWordDict[nextState] + " "
			currState = nextState

		return fakeText