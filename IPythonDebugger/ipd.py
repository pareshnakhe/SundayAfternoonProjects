from IPython.terminal.embed import InteractiveShellEmbed
from IPython.terminal.prompts import Prompts, Token
from traitlets.config.loader import Config
import sys
import inspect


class CustomPrompt(Prompts):
    # some fun with custom prompts

    def in_prompt_tokens(self, cli=None):

        return [
            (Token.Prompt, '<'),
            (Token.PromptNum, 'idb-in'),
            (Token.Prompt, '>: '),
        ]

    def out_prompt_tokens(self):
        return [
            (Token.OutPrompt, '<'),
            (Token.OutPromptNum, 'idb-out'),
            (Token.OutPrompt, '>: '),
        ]


class IPythonDebugger:
    banner = ">>> Custom IPython Debugger"
    exit_msg = "Leaving debugger"

    def __init__(self, args=[]):

        if args:
            self.exec_cmd = " ".join(self.args)
        else:
            self.exec_cmd = inspect.getframeinfo(
                inspect.currentframe().f_back
            ).filename

        cfg = Config()
        cfg.TerminalInteractiveShell.prompts_class = CustomPrompt

        self.embedded_shell = InteractiveShellEmbed(
            config=cfg,
            banner1=self.banner,
            exit_msg=self.exit_msg
        )
        self.embedded_shell.run_line_magic('matplotlib', 'osx')

    def ipython_debugger(self):

        try:
            get_ipython
        except NameError:
            # print('running outside IPython')
            self.embedded_shell.run_line_magic('run', self.exec_cmd)
            sys.exit(1)
        else:
            # print('running in IPython')
            return self.embedded_shell
