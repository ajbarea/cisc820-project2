# fun2 data

The instructor's ReadMe for function 2, transcribed from the posted `ReadMe.rtf`.

> for b, c & A in function 2
>
> - is saved in corresponding 'fun2_b.txt','fun2_c.txt', and 'fun2_A.txt'
> - b and c should be read out as column vectors, where b is of dimension m by 1, and c is of dimension n by 1
> - A should be read out as matrices of dimension m by n. Each column of A, i.e., A(:,i) corresponds to a_i
> - Examples of how to read the matrices out:
>
> ```matlab
> fid = fopen('fun2_A.txt','r');
> A = fscanf(fid,'%e ',[500,100]);
> fclose(fid);
> ```
>
> - The gradient and Hessian of function 2 is given in project description. Hint: you should use vector & matrix calculation in your code as much as possible, instead of accessing to the elements in the vectors or matrices. The matlab expression of the function, its gradient and Hessian is basically given in the project description.

## In Python

`fscanf` fills the 500 x 100 matrix column by column, so in NumPy:

```python
A = np.loadtxt("fun2_A.txt").reshape((500, 100), order="F")
b = np.loadtxt("fun2_b.txt")  # (500,)
c = np.loadtxt("fun2_c.txt")  # (100,)
```

`functions.load_fun2()` already does this.
