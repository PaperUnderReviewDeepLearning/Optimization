from arguments import get_args
from solver_cbs import CBSSolver
from solver_base import BaseSolver
def main():
    args = get_args()

    solver = CBSSolver(args)

    print('training!')
    solver.solve()
    print('done')


    if args.save_model:
        solver.save_model()

if __name__ == '__main__':
    main()
