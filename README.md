# Sudoku-Solver
Feed an image of a sudoku problem and watch it get solved
## How It Works
1. **Upload** a photo of any Sudoku puzzle
2. The app **extracts** the grid using computer vision (OpenCV)
3. **Digits are recognized** using a structural hole topology classifier + KNN 
4. Hit **Solve** then the answer appears

## Running the app 
In terminal run 
```bash
python3 sudoku_solver.py
```
or in VS code (or other IDE) hit the run button 

## Things to know
1. Sometimes when loading a file it will be imposible to solve 
2. Test4 loads the probelm in worng but it is still solviable 
3. You can not edit the puzzle once its loaded in 
