
import re
import functools
import nltk
import os
from pathlib import Path
from typing import Optional
from stextools.snify.text_anno.local_stex_catalog import (local_flams_stex_catalogs, LocalFlamsCatalog,)
from stextools.snify.annotype import AnnoType, StateType, StepperStatus
from stextools.snify.objective_anno.objective_anno_state import ObjectiveAnnoState
from stextools.snify.snify_commands import SkipCommand
from stextools.stepper.command import CommandCollection, Command, CommandInfo, CommandOutcome
from stextools.stepper.document import Document, STeXDocument
from stextools.stepper.document_stepper import TextRewriteOutcome, SubstitutionOutcome
from stextools.stepper.interface import interface
from stextools.stepper.stepper import Modification
from stextools.stepper.stepper_extensions import QuitCommand, UndoCommand, RedoCommand
from stextools.stex.flams import FLAMS
from stextools.utils.json_iter import json_iter

#print(dir(interface))

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\ai-agents\source\mod\search-based-agent.en.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\cgroup.zhs.tex"
 
# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\field.en.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\field.de.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\field.zhs.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\ideal.en.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\ideal.de.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\inverse.en.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\inverse.de.tex"


# add
original_get_content= Document.get_content
#monkey-patching 
def _get_content_utf8_safe(self):
    if self._content is None:
        self._content=self.path.read_text(encoding= "utf-8")
    return self._content
Document.get_content= _get_content_utf8_safe

DEFAULT_LANGUAGE = "en"
Language_pos: dict[str, dict[str, Optional[str]]] = {"en": {"TO":None, "IN": None, "JJ": ":A", "RB": ":A", "NN":":N", "NNS":":N", "VB":":V", "VBD":":V",},
                                                     "de":{"ADP": None, "ADJ": ":A", "ADV": ":A", "NOUN": ":N", "PROPN": ":N", "VERB": ":V", "AUX":":V",},
                                                       } #a dictionary of dictionaries of tags in diffrent languages
Language_prepositions: dict[str, list[str]]= {"en": ["for", "from", "at", "of", "to","with"], "de":["für", "von", "bei", "zu", "mit","auf"],}


@functools.cache
def get_stex_catalogs() -> dict[str, LocalFlamsCatalog]:
    return local_flams_stex_catalogs()


def get_document_language(document: Document)->str:
    """determine the language of the document"""
    lang= getattr(document, "lang", None)
    if lang:
        return lang

    path= str(getattr(document, "path", " "))
    match= re.search(r"\.([a-z]{2,3})\.tex$", path)
    if match:
        return match.group(1)
    
    return DEFAULT_LANGUAGE


def pos_tag_for_language(tokens: list[str], language:str)-> list[tuple[str, str]]:
    #tokenize and tag words for the given language
    if language=="en":
        return nltk.pos_tag(tokens)
    try:
        import spacy
        model_name= {"de": "de_core_news_sm",
                     "zhs": "zhs_core_news_sm",}.get(language)
        if model_name is None:
            raise OSError(f"no confingurated spaCy model for lang={language!r}")

        nlp= get_spacy_pipeline(model_name)
        Doc= nlp(" ".join(tokens))
        return [(t.text, t.pos_) for t in Doc] # extract (word, tag) for each token
    except Exception as e:
        interface.write_text( f"\nno part of speech tagger available for "
                             f"lang={language!r} ({e}); falling back to the english tagger, "
                             f"results may be poor. \n")
        return nltk.pos_tag(tokens)

def get_spacy_pipeline(model_name: str):
    import spacy
    return spacy.load( model_name)

def build_suggestions(correspondances: list[str], kind: str, line: str, symbol_name:str, num_args: int, form: Optional[str], source_language:str, language: str, displayed_name:str, document:str, source_document_path:str)-> list[str]:
    form= form or ""
    if not language:
        language= DEFAULT_LANGUAGE

    pos_map= Language_pos.get( language, Language_pos[DEFAULT_LANGUAGE])
    prepositions= Language_prepositions.get(language, Language_prepositions[DEFAULT_LANGUAGE])

    prep=""
    match_name= None
    tags_name: list[tuple[str, str]] = []
    suggestions: list[str]= []
    if kind=="definiendum":
        match=re.search(r'\\definiendum\{([^}]*)\}\{([^}]*)\}',document, re.DOTALL)
        match_name=re.sub(r'\s+', ' ', match.group(1)).strip()
        
        if displayed_name:
            tokens_name= nltk.word_tokenize(displayed_name)
            tags_name=pos_tag_for_language(tokens_name, language)
            
            converted=[]
            for word, pos in tags_name:
                tag= pos_map.get(pos, " ")
                if tag is None:
                    continue
                converted.append((word, tag))
            
            #suggestion 1
            suggestion = "#1"
            for word,tag in converted:
                suggestion += f' {word }{tag}'       
            suggestions.append(suggestion.strip())

            #suggestion 2
            suggestion=""

            for word,tag in converted:
                suggestion += f'{word }{tag} '
            
            for word, pos_name in tags_name:
                if pos_name in ("ADP", "IN", "TO") : 
                    suggestion+= f'{word} '
                    break
            if prep and not any(pos_name in ("ADP", "IN", "TO") for _, pos_name in tags_name ):
                    suggestion+= "" +prep+ ""
            suggestion+="#1"

            if suggestion.strip() not in suggestions:
                suggestions.append(suggestion.strip())

            #suggestion 3
            suggestion= "#1 "
            for word, tag in converted:
                suggestion += f"{word}{tag} "
    
            for word, pos_name in tags_name:
                if pos_name in ("ADP", "IN", "TO") :
                    suggestion+= f"{word} "
            if prep:
                suggestion+= prep + " "
            suggestion+='#2'
            if suggestion.strip() not in suggestions:
                suggestions.append(suggestion.strip())
            
    
            #suggestion 4
            suggestion = "#2 "
            for word, tag in converted:
                suggestion += f"{word}{tag} "
            for word, pos_name in tags_name:
                if pos_name in ("ADP", "IN", "TO") :
                    suggestion+= f"{word} "
            if prep:
                suggestion+= prep + " "
            suggestion+= "#1"

            if suggestion.strip() not in suggestions:
                suggestions.append(suggestion.strip())

    if kind== "symdecl":
        if displayed_name:
            tokens_name= nltk.word_tokenize(displayed_name)
            tags_name=pos_tag_for_language(tokens_name, language)
            
            converted=[]
            for word, pos in tags_name:
                tag= pos_map.get(pos, " ")
                if tag is None:
                    continue
                converted.append((word, tag))
                    
        match= re.search(r'\\symdecl\*?\{([^}]*)\}', document)
        if not match:
            return None

        displayed_name= match.group(1)
        #suggestion 1
        suggestion = "#1"
        for word,tag in converted:
            suggestion += f' {word }{tag}'       
        suggestions.append(suggestion.strip())

        #suggestion 2
        suggestion=""

        for word,tag in converted:
            suggestion += f'{word }{tag} '
        
        for word, pos_name in tags_name:
            if pos_name in ("ADP", "IN", "TO") : 
                suggestion+= f'{word} '
            else:
                suggestion+= "" +prep+ ""
        suggestion+="#1"
        if suggestion.strip() not in suggestions:
            suggestions.append(suggestion.strip())

        #suggestion 3
        suggestion= "#1 "
        for word, tag in converted:
            suggestion += f"{word}{tag} "

        for word, pos_name in tags_name:
            if pos_name in ("TO", "IN", "ADP", "SCONJ"):
                suggestion+= f"{word} "
        if prep:
            suggestion+= prep + " "
        suggestion+='#2'
        if suggestion.strip() not in suggestions:
            suggestions.append(suggestion.strip())
        

        #suggestion 4
        suggestion = "#2 "
        for word, tag in converted:
            suggestion += f"{word}{tag} "
        for word, pos_name in tags_name:
            if pos_name in ("TO", "IN", "ADP", "SCONJ"):
                suggestion+= f"{word} "
        if prep:
            suggestion+= prep + " "
        suggestion+= "#1"

        if suggestion.strip() not in suggestions:
            suggestions.append(suggestion.strip())
        
#just to have the prepositions
    elif kind== "symdef" or kind=="symdecl":
        match_name= re.search(r"name= ([^,\]]+)", line)
        if match_name:
            tokens_name= [match_name.group(1)]
            tags_name= pos_tag_for_language(tokens_name, language)
        else:
            for p in prepositions:
                if symbol_name.endswith(p): 
                    prep= p
                    break
#generation of suggeestions
   
    for correspondance in correspondances:
        tokens= nltk.word_tokenize(correspondance)
        tags= pos_tag_for_language(tokens, source_language)

        converted=[]
        for word, pos in tags:
            tag= pos_map.get(pos, " ")
            if tag is None:
                continue
            converted.append((word, tag))   

        suggestion= ""
        if num_args==1:
            #pos= form.find('{#1}')
            if (form.startswith('{#1}') or form.startswith('#1')):
                suggestion += "#1"
                for word,tag in converted:
                    suggestion += f' {word }{tag}'       
                suggestions.append(suggestion.strip())
            elif (form.endswith('{#1}') or form.endswith('#1')):
                for word,tag in converted:
                    suggestion += f'{word }{tag} '
                if match_name:
                    for word, pos_name in tags_name:
                        if pos_name in ('TO', 'IN'): 
                            suggestion+= f'{word} '
                else:
                    suggestion+= " " +prep+ " "
                suggestion+="#1"
            if suggestion not in suggestions:
                suggestions.append(suggestion.strip())

        elif num_args== 2:
            position1 = form.find ("#1")
            if position1==-1: 
                position1= form.find("{#1}")
            position2= form.find("#2")
            if position2== -1:
                position2= form.find("{#2}")

            args_in_order= position2==-1 or position1==-1 or position1 < position2

            if args_in_order:
                suggestion= "#1 "
                for word, tag in converted:
                    suggestion += f"{word}{tag} "

                if match_name: 
                    for word, pos_name in tags_name:
                        if pos_name in ("TO", "IN", "ADP", "SCONJ"):
                            suggestion+= f"{word} "
                elif prep:
                    suggestion+= prep + " "
                suggestion+='#2'
            else: 
                suggestion = "#2"
                for word, tag in converted:
                    suggestion += f"{word}{tag} "

                if match_name:
                    for word, pos_name in tags_name:
                        if pos_name in ("TO", "IN", "ADP", "SCONJ"):
                            suggestion+= f"{word} "
                elif prep:
                    suggestion+= prep + " "
                suggestion+= "#1"
            if suggestion not in suggestions:
                suggestions.append(suggestion.strip())

        elif num_args==3:
            suggestion= "#1 "
            for word,tag in converted:
                suggestion+= f'{word }{tag} '
            suggestion += "from #2 to #3"
            if suggestion not in suggestions:
                suggestions.append(suggestion.strip())

    return suggestions

class EnglishDocSymbol:
    def __init__( self, uri:str, path:str):
        self.uri= uri
        self.path=path

    def english_symbol_from_not_english_document(self, source_document_path: str, symbol_name:str,):
        source_document_path=str(source_document_path)
        match= re.match(r"^(.*)?.[a-z]{2,3}\.tex$", source_document_path)
        if not match:
            return None

        english_path= f"{match.group(1)}.en.tex"
        if not os.path.isfile(english_path):
            return None
        with open(english_path, encoding= "utf8") as f:
            english_text=f.read()
       
        sym_match= re.search(
            rf'\\symdef\*?\{{{re.escape(symbol_name)}\}}(?:\[[^\]]*\])?\{{.*\}}', english_text,
        )
        kind="symdef"
        if sym_match is None:
            sym_match= re.search(rf'\\symdecl\*?\{{{re.escape(symbol_name)}\}}', english_text,)
            kind= "symdecl"

        if sym_match is None:
            return None
        uri= self.get_uri_from_annotations( english_text, sym_match.start(). kind, file_path= english_path,)
        if uri is None:
            return None
        return EnglishDocSymbol(kind= kind, uri=uri, path= english_path, declaration= sym_match.group(0))


class VerbalizationAnnoState:
    pass


class AddVerbalizationCommand(Command):
    def __init__(self, insert_position:int,  symbol_name:str, document_content:str, uri:str, num_args: int, suggestions:list[str], lang:str ):
        self.insert_position = insert_position
        self.symbol_name= symbol_name
        self.document_content = document_content
        self.uri= uri
        self.num_args= num_args
        self.suggestions= suggestions
        self.lang= lang
        #self.snify_state= snify_state

        super().__init__(CommandInfo(
            pattern_presentation='a',
            description_short='dd verbalization',
            description_long='Add a verbalization for the current symbol.'
        ))

    def specifications(self)-> Optional[str]:
        if self.suggestions:
            interface.write_text("\n You can choose a verbalization or write your own:\n")
            for i, suggestion in enumerate(self.suggestions, start=1):
                interface.write_text(f"{i}) {suggestion}\n")

            answer= interface.get_input()

            try:
                choice= int(answer)
            except ValueError: #the user has typed his own verbalization directly
                return answer

            if not (1<= choice<=len(self.suggestions)):
                interface.write_text("Invalid suggestion number.\n")
                return None

            specifications= self.suggestions[choice-1]
            interface.write_text(f"\n Selected suggestion:\n{specifications}\n" "press enter to accept or type a modified version:\n")
            modified= interface.get_input().strip()
            return modified if modified else specifications
        
        interface.write_text("\n Please enter the verbalization:\n")
        return interface.get_input()


    def annotation_type( self, specifications:str)-> Optional[str]:
        num_arguments= len(re.findall(r'#\d+', specifications))
        colon_pos = [i for i, char in enumerate(specifications)
               if char==':']
        if not colon_pos:
            interface.write_text("No annotation tag found.\n")
            return None

        letter= specifications[colon_pos[-1]+1]
        annotation_type= letter.upper() if num_arguments<= 1 else letter.upper() +str(num_arguments)

        interface.write_text(f"Annotation type [{annotation_type}] (Enter to accept): ")
        answer = interface.get_input()
        return answer if answer else annotation_type

    def arg_types(self, num_arguments: int)-> str:
        args= ""
        #doc_language= get_document_language(self.document_content)
        if self.lang=="en":
            while num_arguments >0:
                interface.write_text("\n Argument types (c/d)? Press enter to skip:\n")
                entered= interface.get_input().strip()
                if entered== "":
                    break
                if len( entered)== num_arguments and all(ch in "cd" for ch in entered):
                    args= entered
                    break
                interface.write_text (f"Please enter exactly {num_arguments} characters consisting of c and /or d.\n")
            #return args
        elif self.lang=="de":
            while num_arguments >0:
                interface.write_text("\n Argument types (k/d)? Press enter to skip:\n")
                entered= interface.get_input().strip()
                if entered== "":
                    break
                if len( entered)== num_arguments and all(ch in "kd" for ch in entered):
                    args= entered
                    break
                interface.write_text (f"Please enter exactly {num_arguments} characters consisting of k and /or d.\n")
        return args


    def execute(self,  call: str) -> list[CommandOutcome]:
        """ this is called when the user presses 'a' """
        
        specifications= self.specifications()
        if not specifications:
            return []

        annotation_type= self.annotation_type( specifications)
        if annotation_type is None:
            return []
        
        num_arguments= len(re.findall(r'#\d+', specifications))
        spec= re.sub(r'#\d+', '', specifications)
        spec= re.sub(r':[A-Za-z]+', '', spec)
        words= spec.split()
        name = "-".join(words)

        args= self.arg_types(num_arguments)

        optional= f"[Name= {name}, args= {args}]" if args else f"[name= {name}]"

        new_verbalization= (f'\\verbalization{{{self.symbol_name}}}{optional}{{{annotation_type}}}{{{specifications}}}\n')
        if new_verbalization in self.document_content:
            interface.write_text( "this verbalization already exist.\n")
            return []

        return [SubstitutionOutcome(new_verbalization, self.insert_position, self.insert_position)]

#  python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\divgroup.en.tex"

#  python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\ai-agents\source\mod\goal-based-agent.en.tex"
class DeleteVerbalizationCommand(Command):

    def __init__(self, position: int, symbol_name:str, document_content:str):
        self.position= position
        self.symbol_name=symbol_name
        self.document_content=document_content
        super().__init__(CommandInfo(
            pattern_presentation='d',
            description_short='elete verbalization',
            description_long='Delete a verbalization for the current symbol.'
        ))

    def execute(self, call:str)-> list[CommandOutcome]:
        """this is called when the user presses 'd' """
        pattern= rf'\\verbalization\{{{re.escape(self.symbol_name)}\}}\[.*?\]\{{.*?\}}\{{.*?\}}'
        matches= list(re.finditer(pattern, self.document_content))
        if not matches:
            interface.write_text(f'No verbalizations found for "{self.symbol_name}".\n')
            return []
        interface.write_text(f'Existing verbalizations for "{self.symbol_name}": \n')
        for i, match in enumerate(matches, start=1):
            interface.write_text(f"{i}) {match.group(0)}\n")
        interface.write_text("\n which verbalization do you want to delete: \n")
        answer= interface.get_input()
        try: 
            choice=int(answer)
            selected_match=matches[choice-1]
        except(ValueError, IndexError):
            interface.write_text('\n Invalid choice.\n')
            return[]
        start=selected_match.start()
        end= selected_match.end()
        while(
            end>len(self.document_content)
            and self.document_content[end] in '\n\r'
        ):
            end+=1
        return[
            SubstitutionOutcome('', start, end)
        ]
    
class VerbalizationAnnoType(AnnoType[VerbalizationAnnoState]):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return f'verbalization-anno'

    def is_applicable(self, document: Document) -> bool:
        if 'verbalizations' not in self.snify_state.mode:
            return False
        if isinstance(document, STeXDocument):
            return True
        else:
            return False

    def get_initial_state(self) -> StateType:
        get_stex_catalogs()
        return VerbalizationAnnoState()

    def get_uri_from_annotations( self, document_content:str, position: int, kind: str, symbol_name: Optional[str]= None, source_lang: str="en",):
        language= get_document_language(document_content)
        print(language)
        if kind== "definiendum":
            
            #looks in the whole catalog
            catalog= get_stex_catalogs()[language]
            #print(catalog.symb_to_verb)
            for symbol in catalog.symb_iter():
                
                if symbol.uri.endswith(f"s={symbol_name}"):
                    return symbol
            #looks for the symdef/-decl from the english version of the document
            current_path= str(self.snify_state.get_current_document().path)
            english_symbol= self.english_symbol_from_not_english_document(current_path, symbol_name)

            if english_symbol is None:
                return english_symbol
            return None

        line_no = document_content[:position].count("\n")
        annotations= FLAMS.get_file_annotations(str(self.snify_state.get_current_document().path), load=True,)
    
        for e in json_iter(annotations):
            if not isinstance(e, dict):
                continue
            if kind== "symdef" and "Symdef" in e:
                if e["Symdef"]["uri"]["range"]["start"]["line"]== line_no:
                    return e["Symdef"]["uri"]["uri"]
            if kind== "symdecl" and "Symdecl" in e:
                if e["Symdecl"]["uri"]["range"]["start"]["line"]== line_no:
                    return e["Symdecl"]["uri"]["uri"]        

        return None

    def find_next_definiendum_position(self, document_content:str, symbol_name:str,)->Optional[int]:
        pattern=rf'\\definiendum\{{{re.escape(symbol_name)}\}}'
        match= re.search(pattern, document_content)
        if match:
            return match.start()
        return None
    
    def extract_symbol_information(self, document_content: str, position: int, source_lang:str="en"):
        remaining_document = document_content[position:]

        form=None
        num_args= 0

        #definiendum
        if "\\definiendum" in remaining_document:
            match= re.search(r'\\definiendum\{([^}]*)\}\{([^}]*)\}', remaining_document, re.DOTALL)

            kind= "definiendum"
            symbol_name = re.sub(r'\s+', ' ', match.group(1)).strip()
            displayed_name= re.sub(r'\s+', ' ',match.group(2)).strip()

            english_symbol= self.get_uri_from_annotations( document_content, position, kind, symbol_name, source_lang=source_lang,)
            if english_symbol is None:
                return None
            uri= english_symbol.uri
            with open(english_symbol.path, encoding= "utf8") as f:
                english_text= f.read()

            symdef_match= re.search(rf'\\symdef\{{{re.escape(symbol_name)}\}}(?:\[[^\]]*\])?\{{(.*)\}}', english_text,)

            if symdef_match:
                symdef_line= symdef_match.group(0)
                args_match= re.search(r'args=(\d+)', symdef_line)
                if args_match:
                    num_args= int(args_match.group(1))

                formula_match= re.search(r'\\symdef\{[^}]+\}(?:\[[^\]]*\])?\{(.*)\}$', symdef_line,)

                if formula_match:
                    form= formula_match.group(1)

            else:
                symdef_line=""

            begin_pos= document_content.rfind(r"\begin{sdefinition}", 0, position)
            pattern = r'\\verbalization\{[^}]*\}(?:\[[^\]]*\])?\{[^}]*\}\{[^}]*\}'
            matches = list (re.finditer(pattern, document_content))
            if matches: 
                verbalizations_end= document_content.find("\n", matches[-1].start())
                if verbalizations_end==-1:
                    insert_position= len(document_content)
                else:
                    insert_position=verbalizations_end+1
            
            elif begin_pos==-1 and not matches:
                insert_position = position
            else:
                end_verbalization= document_content.find("\n", begin_pos)
                if end_verbalization==-1:
                    end_verbalization= len(document_content)
                insert_position= end_verbalization +1

            return {"kind": kind, "symbol_name": symbol_name, "displayed_name": displayed_name, "uri": uri, "num_args": num_args, "formula": form, "line": symdef_line, "insert_position": insert_position,}

        elif remaining_document.startswith("\\symdef"):
            kind= "symdef"
            match= re.search(r'\\symdef\{([^}]*)\}', remaining_document)
            if not match:
                return None
            displayed_name= match.group(1)

            uri= self.get_uri_from_annotations(document_content, position, kind)
            if uri and "s=" in uri:
                symbol_name= uri.split("s=")[-1] 
            else:
                symbol_name=displayed_name
            args_match= re.search(r'args=(\d+)', remaining_document)

            if args_match:
                num_args= int(args_match.group(1))

            formula_match= re.search(r'\\symdef?\{[^}]+\}(?:\[[^\]]*\])?\{(.*)\}$', remaining_document,)
            if formula_match:
                form= formula_match.group(1)

            pattern = rf'\\verbalization\{{{re.escape(symbol_name)}\}}\[.*?\]\{{.*?\}}\{{.*?\}}'
            matches = list (re.finditer(pattern, document_content))
            if matches: 
                verbalizations_end= document_content.find("\n", matches[-1].start())
                if verbalizations_end==-1:
                    insert_position= len(document_content)
                else:
                    insert_position=verbalizations_end+1

            else:
                line_end= document_content.find("\n", position)
                insert_position= len(document_content) if line_end==-1 else line_end+1

            return {"kind": kind, "symbol_name": symbol_name, "displayed_name": displayed_name, "uri": uri, "num_args": num_args, "formula": form, "line": remaining_document, "insert_position": insert_position,}

        elif remaining_document.startswith("\\symdecl"):
            kind= "symdecl"
            match= re.search(r'\\symdecl\*?\{([^}]*)\}', remaining_document)
            if not match:
                return None

            displayed_name= match.group(1)
            uri= self.get_uri_from_annotations(document_content, position, kind)
            if uri and "s=" in uri:
                symbol_name= uri.split("s=")[-1] 
            else:
                symbol_name=displayed_name
            
            pattern = rf'\\verbalization\{{{re.escape(symbol_name)}\}}\[.*?\]\{{.*?\}}\{{.*?\}}'
            matches = list (re.finditer(pattern, document_content))
            #order the verbalizations if there are already existing verbalization for a symbol
            if matches: 
                verbalizations_end= document_content.find("\n", matches[-1].start())
                if verbalizations_end==-1:
                    insert_position= len(document_content)
                else:
                    insert_position=verbalizations_end+1
            
            else:
                line_end= document_content.find("\n", position)
                insert_position= len(document_content) if line_end==-1 else line_end+1

            return {"kind": kind, "symbol_name": symbol_name, "displayed_name": displayed_name, "uri": uri, "num_args": 0, "formula": None, "line":remaining_document, "insert_position": insert_position,}
        return None

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\algebra.en.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\algebraic.de.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\divgroup.de.tex"

# python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\algebra\source\mod\group.de.tex"

    def get_next_annotation_suggestion(
            self, document: Document, position: int
    ) -> Optional[tuple[int, list[Modification]]]:
        # a string with the content of the file
        document_content = document.get_content() #full document
        # we only care about stuff after the current position
        language= get_document_language(document)
        document_content = document_content[position:] #search content
        if language== "en":
            target_patterns= (r'\symdef', r'\symdecl')
            #positions = [p for p in (document_content.find('\\symdef'), document_content.find('\\symdecl'),) if p!=-1]
        else:
            target_patterns= (r'\definiendum',)
        positions= []
        for target in target_patterns:
            p= document_content.find(target)
            while p!=- 1:
                positions.append(p)
                p= document_content.find(target, p+1)
        if not positions: 
            return None
        #interface.write_text(f"DEBUG: gefunden Kandidaten ( relativ) = {positions}\n")
        for relative_pos in sorted(positions):
            next_pos= position+relative_pos #position in the full document
            info=self.extract_symbol_information(document.get_content(), next_pos) 
            if info is not None:
                return next_pos, []
        #next_pos= position+ min(positions)
            
        return None
            
        
    
        #our_position = min(positions)
        #if our_position == -1:    # we did not find anything
            #return None
        #elif(document_content.find('\\symdef')==-1 and document_content.find('\\symdecl')>=0 ):
         #   our_position = document_content.find('\\symdecl')
        #elif(document_content.find('\\symdef')>=0 and document_content.find('\\symdecl')==-1 ):
         #   our_position = document_content.find('\\symdef')
            
        #return our_position + position, []

    def show_current_state(self):
        interface.clear()
        interface.write_text('\nHELLO, I AM THE VERBALIZATION ASSISTANT\n')

        document_content = self.snify_state.get_current_document().get_content()
        position = self.snify_state.cursor.in_doc_pos
        
        info = self.extract_symbol_information(document_content, position)
        
        if info is None:
            return
        symbol_name= info["symbol_name"]
        line= info["line"]
        kind= info["kind"]

        if kind=="symdef":
            interface.write_text('\nCurrent \\symdef:\n\n')
            
        elif kind=="symdecl":
            interface.write_text('\nCurrent \\symdecl:\n\n')
        else:
            interface. write_text('\nCurrent \\definiendum:\n\n')
        interface.show_code(line, format='sTeX')

    
        pattern = rf'\\verbalization\{{{re.escape(symbol_name)}\}}\[.*?\]\{{.*?\}}\{{.*?\}}'
        matches = list (re.finditer(pattern, document_content))
        if not matches:
            interface.write_text(f'\nNo Verbalizations found for "{symbol_name}".\n'
                    )
            return []
        
        interface.write_text(f'\n Existing verbalizations for "{symbol_name}": \n')

        for i, match in enumerate(matches, start =1):
            interface.write_text(f"{i}) {match.group(0)}\n"         
            )    
        
#  python -m stextools snify --mode=text,verbalizations "C:\Users\ivana\Desktop\MathHub\smglom\calculus\source\mod\derivative.en.tex"

    def get_command_collection(self, stepper_status: StepperStatus) -> CommandCollection:
        position = self.snify_state.cursor.in_doc_pos
        document_content = self.snify_state.get_current_document().get_content()
        #string = document_content[position:]
        info= self.extract_symbol_information(document_content, position)
  
        symbol_name= info["symbol_name"]
        displayed_name= info["displayed_name"]
        line= info["line"]
        kind= info["kind"]
        uri= info["uri"]
        num_args= info.get("num_args", 0)
        form= info.get("formula")
        insert_position= info["insert_position"]
        language= get_document_language(document=self.snify_state.get_current_document())
        source_document_path= self.snify_state.get_current_document().path
        source_lang= "en"
        catalog= get_stex_catalogs()[source_lang]

        correspondances=[]
        for symbol in catalog.symb_iter():
            if symbol.uri == uri:
                for verbalization in catalog.symb_to_verb[symbol]:    
                    if verbalization.verb not in correspondances:
                        correspondances.append(verbalization.verb.replace('\n', ' '))      
           
        suggestions=build_suggestions(correspondances, kind, line, symbol_name, num_args, form, source_lang, language, displayed_name, document_content, source_document_path)

        
        #AddVerbalizationCommand(insert_position, symbol_name, document_content, uri, num_args, suggestions, target_lang)
        #position = position + 1 + string.find('\n')

        return CommandCollection(
            f'snify:{self.name}',
            [
                QuitCommand(),
                UndoCommand(is_possible=stepper_status.can_undo),
                RedoCommand(is_possible=stepper_status.can_redo),
                SkipCommand(self.snify_state, description_short='kip'),
                AddVerbalizationCommand(insert_position, symbol_name, document_content, uri, num_args, suggestions, language),
                DeleteVerbalizationCommand(position, symbol_name, document_content),
            ],
            have_help=True,
        )
