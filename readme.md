# Protein Methylation Prediction

This repo include data normalization, Bi-directional RNN implemented by tensorflow-1.1.0.  
And it is a sequence classification task about protein methylation prediction.

### Max-min normalization
```python
	min_v = np.min(self.train_prop1_data)
	max_v = np.max(self.train_prop1_data)
	self.train_prop1_data = (self.train_prop1_data - min_v) / (max_v - min_v)
	
	with open('../cache/maxmin_prob.txt', 'w') as file:
		file.write(str(min_v) + " " + str(max_v) + "\n")
	
```

### Z-score normalization
```python
	mu = np.mean(self.train_prop1_data)
	sigma = np.std(self.train_prop1_data)
	self.train_prop1_data = (self.train_prop1_data - mu) / sigma
```