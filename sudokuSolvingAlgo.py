def findVacancy(grid):
    for row in range(0,9):
        for col in range(0,9):
            if(grid[row][col]==0):
                return row,col
    return None

def isOkay(grid, row, col, num):
    if grid[row][col]!=0:
        return False

    for i in range(0,9):
        if grid[i][col]==num:
            return False
    
    for j in range(0,9):
        if grid[row][j]==num:
            return False
    
    for i in range( (int(row/3))*3, (int(row/3)+1)*3 ):
        for j in range( (int(col/3))*3, (int(col/3)+1)*3 ):
            if grid[i][j]==num:
                return False

    return True

def sudoku(grid):
    find = findVacancy(grid)
    if not find:
        return True
    else:
        row, col = find
    for num in range(1,10):
        if(isOkay(grid,row,col,num)):
            grid[row][col]=num
            if sudoku(grid):
                return True
            grid[row][col]=0
    return False