from tree_sitter import Language, Parser
import tree_sitter_c_sharp
import tree_sitter_java

def extract_code_chunks(code):
    return extract_methods(code)

def parse_tree(code):
    CS_LANGUAGE = Language(tree_sitter_c_sharp.language())
    parser = Parser(CS_LANGUAGE)
    
    tree = parser.parse(bytes(code,"utf8"))
    return tree

def extract_import_statements(root_node, code):
    usings = ''
    for node in root_node.children:
        if node.type == 'using_directive':
            for child in node.children:
                if child.type == 'qualified_name':
                    usings += code[child.start_byte:child.end_byte] + "\n"
    return usings

def extract_methods(code):
    tree = parse_tree(code)
    root_node = tree.root_node

    usings = extract_import_statements(root_node, code)
    methods = []
    def traverse(node, namespace = ''):
        if node.type == 'namespace_declaration':
            for child in node.children:
                if child.type == 'qualified_name':
                    namespace = f"namespace {code[child.start_byte:child.end_byte]}"
        if node.type == 'method_declaration':
            method_code = code[node.start_byte:node.end_byte]
            methods.append(usings + "\n" + namespace + "\n" + method_code)
            print(usings + "\n" + namespace + "\n" + method_code)
            print("-----------")
        for child in node.children:
            traverse(child, namespace)

    # Traverse the tree starting from the root node
    traverse(root_node)
    if len(methods) == 0:
        return [code]
    else:
        return methods