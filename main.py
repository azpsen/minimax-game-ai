##
## LAB 5 -- CS 4222/5222 -- Fall 2023
##
## 1. You need to implement minimax search
##
## 2. You need to implement alpha-beta pruning with and without cutoff
##
## 3. Try all algorithms with Tic-Tac-Toe
## 4. Try Random and Alpha-Beta cutoff with ConnectFour


import random
from tkinter import *

infinity = 1.0e400


## MiniMax decision (figure 5.3)
def minimax_decision(state, game):
    player = game.to_move(state)

    def max_value(state):
        ### ... you fill this in ...
        ### Hint: you can make use of
        ###   game.terminal_test(state)
        ###   game.utility(state,player)
        ###   game.sucessors(state)

        if game.terminal_test(state):
            return game.utility(state, player)

        v = -infinity
        for a, s in game.successors(state):
            v = max(v, min_value(s))
        return v

    def min_value(state):
        ### ... you fill this in ...

        if game.terminal_test(state):
            return game.utility(state, player)

        v = infinity
        for a, s in game.successors(state):
            v = min(v, max_value(s))
        return v

    action, state = argmax(game.successors(state), lambda x: min_value(x[1]))
    return action


##
## MiniMax with Alpha-Beta pruning (figure 5.7)
## 
def alphabeta(state, game):
    player = game.to_move(state)

    def max_value(state, alpha, beta):
        ### ... you fill this in ...
        if game.terminal_test(state):
            return game.utility(state, player)

        v = -infinity
        for a, s in game.successors(state):
            v = max(v, min_value(s, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        ### ... you fill this in ...
        if game.terminal_test(state):
            return game.utility(state, player)

        v = infinity
        for a, s in game.successors(state):
            v = min(v, max_value(s, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    action, state = argmax(game.successors(state), lambda x: min_value(x[1], -infinity, infinity))
    return action


##
## MiniMax with Alpha-Beta pruning, cutoff and evaluation (section 5.4.2)
##
def alphabeta_cutoff(state, game, d=4, eval_fn=None):
    player = game.to_move(state)
    eval_fn = eval_fn or (lambda state: game.utility(state, player))

    def max_value(state, alpha, beta, depth):
        ### ... you fill this in ...

        # Cutoff test
        if depth >= d:
            return eval_fn(state)

        v = -infinity
        for a, s in game.successors(state):
            v = max(v, min_value(s, alpha, beta, depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v


    def min_value(state, alpha, beta, depth):
        ### ... you fill this in ...

        # Cutoff test
        if depth >= d:
            return eval_fn(state)

        v = infinity
        for a, s in game.successors(state):
            v = min(v, max_value(s, alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    action, state = argmax(game.successors(state), lambda x: min_value(x[1], -infinity, infinity, 0))
    return action


################################################################################
################################################################################
################################################################################

#
# Players for Games
#
from ast import literal_eval


def query_player(game, state):
    "Make a move by querying standard input."
    return literal_eval(input('Your move? '))


def random_player(game, state):
    "A player that chooses a legal move at random."
    return random.choice(game.legal_moves(state))


def alphabeta_player(game, state):
    return alphabeta(state, game)


def play_game(game, *players):
    "Play an n-person, move-alternating game."
    state = game.initial
    while True:
        for player in players:
            move = player(game, state)
            print("moving: " + str(move))
            state = game.make_move(move, state)
            game.display(state)
            print("")
            if game.terminal_test(state):
                return game.utility(state, 'X')


def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    """
    best = seq[0];
    best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best


def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    """
    return argmin(seq, lambda x: -fn(x))


class State:
    """A state has the player to move, a cached utility, a list of moves
    in the form of a list of (x, y) positions, and a board, in the
    form of a dict of {(x, y): Player} entries, where Player is 'X' or
    'O'.  We also store a "winning move" for winning states which contains
    the position where the win was made. 
    """

    def __init__(self, to_move, utility, board, moves, winning_move=None):
        self.to_move = to_move
        self.utility = utility
        self.board = board
        self.moves = moves
        self.winning_move = winning_move


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement
    legal_moves, make_move, utility, and terminal_test. You may
    override display and successors or you can inherit their default
    methods. You will also need to set the .initial attribute to the
    initial state; this can be done in the constructor."""

    def legal_moves(self, state):
        "Return a list of the allowable moves at this point."
        pass  # abstract

    def make_move(self, move, state):
        "Return the state that results from making a move from a state."
        pass  # abstract

    def utility(self, state, player):
        "Return the value of this final state to player."
        pass  # abstract

    def terminal_test(self, state):
        "Return True if this is a final state for the game."
        return not self.legal_moves(state)

    def to_move(self, state):
        "Return the player whose move it is in this state."
        return state.to_move

    def display(self, state):
        "Print or otherwise display the state."
        print(state)

    def successors(self, state):
        "Return a list of legal (move, state) pairs."
        return [(move, self.make_move(move, state))
                for move in self.legal_moves(state)]

    def __repr__(self):
        return '<%s>' % self.__class__.__name__


class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = State(to_move='X', utility=0, board={}, moves=moves)

    def legal_moves(self, state):
        "Legal moves are any square not yet taken."
        return state.moves

    def make_move(self, move, state):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy();
        board[move] = state.to_move
        moves = list(state.moves);
        moves.remove(move)
        utility = self.compute_utility(board, move, state.to_move)
        return State(to_move='O' if state.to_move == 'X' else 'X',
                     utility=utility, board=board, moves=moves,
                     winning_move=move if utility != 0 else None)

    def utility(self, state, player):
        "Return the value to X; 1 for win, -1 for loss, 0 otherwise."
        return state.utility

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end='')
            print('')

    def compute_utility(self, board, move, player):
        "If X wins with this move, return 1; if O return -1; else return 0."
        if len(self.k_in_row(board, move, player, (0, 1))) >= self.k: return 1 if player == 'X' else -1
        if len(self.k_in_row(board, move, player, (1, 0))) >= self.k: return 1 if player == 'X' else -1
        if len(self.k_in_row(board, move, player, (1, -1))) >= self.k: return 1 if player == 'X' else -1
        if len(self.k_in_row(board, move, player, (1, 1))) >= self.k:
            return 1 if player == 'X' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta):
        """If there is a line of k through move on board for player, return
        it. Otherwise, return empty list
        """
        x, y = move
        delta_x, delta_y = delta
        line = []
        while board.get((x, y)) == player:
            line.append((x, y))
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            line.append((x, y))
            x, y = x - delta_x, y - delta_y
        # Because we added move itself twice
        line.remove(move)
        if len(line) >= self.k:
            return line
        else:
            return []


class ConnectFour(TicTacToe):
    """A TicTacToe-like game in which you can only make a move on the bottom
    row, or in a square directly above an occupied square.  Traditionally
    played on a 7x6 board and requiring 4 in a row."""

    def __init__(self, h=7, v=6, k=4):
        TicTacToe.__init__(self, h, v, k)

    def legal_moves(self, state):
        "Legal moves are any square not yet taken."
        return [(x, y) for (x, y) in state.moves
                if y == self.v or (x, y + 1) in state.board]


class GameInterface:
    def __init__(self, game, title="Default", geometry="750x500"):
        self.game = game
        self.currentState = self.game.initial

        ## Initialize environemnt
        self.root = Tk()
        self.root.title(title)
        self.root.geometry(geometry)

        ## Set up canvas for input window
        self.canvas = Canvas(self.root, width=500, height=500)
        self.canvas.place(x=0, y=0)

        ## Bind mouse clicks to canvas
        self.canvas.bind("<Button-1>", self.click)
        # self.canvas.bind("<Button-3>", self.clear)
        self.waiting = BooleanVar()
        self.waiting.set(1)
        self.move = None

        ## set up radio button: opponent type, search depth
        self.control = Frame(self.root)
        self.control.place(x=510, y=10)

        Label(self.control, width=13, text="Computer play").pack(anchor=W)
        self.controlFrame1 = Frame(self.control, height=2, bd=1, relief=SUNKEN)
        self.controlFrame1.pack(fill=X, padx=5, pady=5)
        self.mode = IntVar()
        self.mode.set(1)
        self.depth = IntVar()
        self.depth.set(4)

        self.randButton = Radiobutton(self.controlFrame1, text="Random", variable=self.mode, value=1).pack(anchor=W)
        self.minimaxButton = Radiobutton(self.controlFrame1, text="MiniMax", variable=self.mode, value=2).pack(anchor=W)
        self.alphaBetaButton = Radiobutton(self.controlFrame1, text="AlphaBeta", variable=self.mode, value=3).pack(
            anchor=W)
        self.alphaBetaCutoffButton = Radiobutton(self.controlFrame1, text="AlphaBeta with cutoff", variable=self.mode,
                                                 value=4).pack(anchor=W)

        self.controlFrame2 = Frame(self.control, height=2, bd=1)
        self.controlFrame2.pack(fill=X, padx=5, pady=10)
        Label(self.controlFrame2, text="     Search tree depth").pack(side="left")
        self.depthControl = Spinbox(self.controlFrame2, from_=1, to=10, width=2, textvariable=self.depth,
                                    state=DISABLED)
        self.depthControl.pack(side="left")

        ## Disable depth spinbox if cutoff is selected
        self.mode.trace("w", lambda name, index, mode: self.depthControl.configure(
            state=['normal' if self.mode.get() == 4 else 'disabled']))

        Button(self.control, text="Reset", command=self.reset).pack(anchor=W, fill=X)
        Button(self.control, text="Exit", command=self.exit).pack(anchor=W, fill=X)

        self.message = StringVar()
        self.message.set("")
        self.messageArea = Label(self.control, font=("Helvetica", 16), textvariable=self.message)
        self.messageArea.pack(fill=X, padx=5, pady=10)

        self.play()

    def draw(self):
        ## Draw the board lines  
        for i in range(0, 500, int(500 / self.game.v)):
            self.canvas.create_line(0, i, 500, i);
        for i in range(0, 500, int(500 / self.game.h)):
            self.canvas.create_line(i, 0, i, 500);

        ## Draw the moves
        board = self.currentState.board
        for x in range(1, self.game.h + 1):
            for y in range(1, self.game.v + 1):
                self.canvas.create_text(((x - 1) * (500 / self.game.h) + (250 / self.game.h),
                                         (y - 1) * (500 / self.game.v) + 250 / self.game.v), text=board.get((x, y), ''),
                                        font=("Helvetica", 44, "bold"));
        self.root.update_idletasks()
        self.root.update()

    def exit(self):
        self.canvas.delete("all")
        self.waiting.set(0)
        self.currentState = self.game.initial
        self.root.destroy()

    def reset(self):
        self.canvas.delete("all")
        self.message.set("")
        self.root.update_idletasks()
        self.root.update()
        self.play()

    def click(self, event):
        if not self.waiting.get(): return
        self.move = (int(event.x / (500 / self.game.h)) + 1, int(event.y / (500 / self.game.v)) + 1)
        print(self.move)
        print(self.currentState.moves)
        self.waiting.set(0)

    def play(self):
        self.currentState = self.game.initial
        while True:

            ## Computer moves
            self.message.set("thinking...")
            self.draw()
            if self.mode.get() == 1:
                move = random.choice(self.game.legal_moves(self.currentState))
            elif self.mode.get() == 2:
                move = minimax_decision(self.currentState, self.game)
            elif self.mode.get() == 3:
                move = alphabeta(self.currentState, self.game)
            elif self.mode.get() == 4:
                move = alphabeta_cutoff(self.currentState, self.game, self.depth.get())
            self.currentState = self.game.make_move(move, self.currentState)
            self.draw()
            if self.game.terminal_test(self.currentState): return self.processEndGame()

            ## wait for player to move
            self.move = None
            while self.move not in self.game.legal_moves(self.currentState):
                self.waiting.set(1)
                self.message.set("waiting for player...")
                self.canvas.wait_variable(self.waiting)

            move = self.move

            self.currentState = self.game.make_move(move, self.currentState)
            self.draw()
            if self.game.terminal_test(self.currentState):
                self.waiting.set(1)
                self.canvas.wait_variable(self.waiting)
                self.processEndGame()
                return

    def processEndGame(self):
        utility = self.game.utility(self.currentState, 'X')
        if (utility == 1): self.message.set("X wins!")
        if (utility == 0): self.message.set("Draw!")
        if (utility == -1): self.message.set("O wins!")

        ## Draw a red line through the winning set of k moves 
        if utility != 0:
            ## Get list of k-in-a-row moves
            board = self.currentState.board
            move = self.currentState.winning_move
            player = board.get(move)
            for orientation in [(0, 1), (1, 0), (1, -1), (1, 1)]:
                line = self.game.k_in_row(board, move, player, orientation)
                if len(line) >= self.game.k: break

            ## find min and max dimensions in the winning move set
            mins = list(map(min, zip(*line)))
            maxs = list(map(max, zip(*line)))
            ## swap y dimensions for falling diagonals
            if (mins[0], mins[1]) not in line:
                mins[1], maxs[1] = maxs[1], mins[1]
            ## draw the line
            self.canvas.create_line((mins[0] - 1) * (500 / self.game.h) + (250 / self.game.h),
                                    (mins[1] - 1) * (500 / self.game.v) + (250 / self.game.v),
                                    (maxs[0] - 1) * (500 / self.game.h) + (250 / self.game.h),
                                    (maxs[1] - 1) * (500 / self.game.v) + (250 / self.game.v), fill="red", width=4)
        self.root.update_idletasks()
        self.root.update()


if __name__ == "__main__":
    ##app = GameInterface(ConnectFour())
    app = GameInterface(TicTacToe())
    app.root.mainloop()
