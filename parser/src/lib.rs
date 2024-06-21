use quote::{quote, ToTokens};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::{fs, io};
use syn::spanned::Spanned;
use syn::{ImplItem, Item, ItemEnum, ItemFn, ItemImpl, ItemStruct};

pub mod github;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TContext {
    pub module: Option<String>,
    pub file_path: Option<String>,
    pub file_name: Option<String>,
    pub struct_name: Option<String>,
    pub snippet: Option<String>,
}

impl TContext {
    fn add_snippet(&mut self, lines: &[&str], line_from: usize, line_to: usize) {
        let mut snippet = String::new();
        for line in &lines[line_from - 1..line_to] {
            snippet.push_str(line);
            snippet.push('\n');
        }
        self.snippet = Some(snippet);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeType {
    Function,
    Struct,
    Enum,
    Impl,
}

use std::fmt;

impl fmt::Display for CodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => write!(f, "Function"),
            Self::Struct => write!(f, "Struct"),
            Self::Enum => write!(f, "Enum"),
            Self::Impl => write!(f, "Impl"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCode {
    pub name: String,
    pub signature: String,
    pub code_type: CodeType,
    pub docstring: Option<String>,
    pub line: usize,
    pub line_from: usize,
    pub line_to: usize,
    pub context: Option<TContext>,
}

pub fn parse_impl(item: &ItemImpl, context: TContext, lines: &[&str]) -> Vec<TCode> {
    let mut functions = Vec::new();
    for item in &item.items {
        if let ImplItem::Fn(method) = item {
            let signature = &method.sig;
            let docstring = method
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("doc"))
                .map(|attr| attr.to_token_stream().to_string());

            let mut context = context.clone();
            let line = method.sig.ident.span().start().line;
            let line_from = method.span().start().line;
            let line_to = method.span().end().line;
            context.add_snippet(lines, line_from, line_to);

            let function = TCode {
                name: method.sig.ident.to_string(),
                signature: quote!(#signature).to_string(),
                code_type: CodeType::Function,
                docstring,
                line,
                line_from,
                line_to,
                context: Some(context.clone()),
            };
            functions.push(function);
        }
    }

    functions
}

pub fn parse_enum(item: &ItemEnum, mut context: TContext, lines: &[&str]) -> TCode {
    let line = item.ident.span().start().line;
    let line_from = item.span().start().line;
    let line_to = item.span().end().line;
    context.add_snippet(lines, line_from, line_to);

    let docstring = item
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("doc"))
        .map(|attr| attr.to_token_stream().to_string());

    TCode {
        name: item.ident.to_string(),
        signature: quote!(#item).to_string(),
        code_type: CodeType::Enum,
        docstring,
        line,
        line_from,
        line_to,
        context: Some(context),
    }
}

pub fn parse_struct(item: &ItemStruct, mut context: TContext, lines: &[&str]) -> TCode {
    let line = item.ident.span().start().line;
    let line_from = item.span().start().line;
    let line_to = item.span().end().line;
    context.add_snippet(lines, line_from, line_to);

    let docstring = item
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("doc"))
        .map(|attr| attr.to_token_stream().to_string());

    TCode {
        name: item.ident.to_string(),
        signature: quote!(#item).to_string(),
        code_type: CodeType::Struct,
        docstring,
        line,
        line_from,
        line_to,
        context: Some(context),
    }
}

pub fn parse_fn(item: &ItemFn, mut context: TContext, lines: &[&str]) -> TCode {
    let signature = &item.sig;
    let docstring = item
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("doc"))
        .map(|attr| attr.to_token_stream().to_string());

    let line = item.sig.ident.span().start().line;

    let line_from = item.span().start().line;
    let line_to = item.span().end().line;
    context.add_snippet(lines, line_from, line_to);

    TCode {
        name: item.sig.ident.to_string(),
        signature: quote!(#signature).to_string(),
        code_type: CodeType::Function,
        docstring,
        line,
        line_from: line,
        line_to: item.block.span().end().line,
        context: Some(context),
    }
}

pub fn parse_item(item: &Item, context: TContext, lines: &[&str]) -> (Vec<TCode>, Vec<TCode>) {
    let mut structs = vec![];
    let mut functions = vec![];
    let mut context = context.clone();

    match item {
        Item::Impl(item) => {
            let impl_block_name = item.self_ty.to_token_stream().to_string();
            context.struct_name = Some(impl_block_name);
            functions.extend(parse_impl(item, context, lines));
        }
        Item::Enum(item) => {
            structs.push(parse_enum(item, context, lines));
        }
        Item::Struct(item) => {
            structs.push(parse_struct(item, context, lines));
        }
        Item::Fn(item) => {
            functions.push(parse_fn(item, context, lines));
        }
        _ => {}
    }

    (functions, structs)
}

// one possible implementation of walking a directory only visiting files
pub fn visit_rs_files(dir: &Path, cb: &mut dyn FnMut(&Path)) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().unwrap() != "target" {
                    visit_rs_files(&path, cb)?;
                }
            } else {
                let path = entry.path();
                if path.extension().unwrap_or_default() == "rs" {
                    cb(&path);
                }
            }
        }
    }
    Ok(())
}

/// Processes a directory.
pub fn process_dir(path: PathBuf) -> Vec<TCode> {
    let mut functions: Vec<TCode> = vec![];
    let mut structs: Vec<TCode> = vec![];

    let dir_path = &path;

    visit_rs_files(dir_path, &mut |path| {
        let relative_path = path.strip_prefix(dir_path).unwrap();

        let file_content = fs::read_to_string(path).unwrap();

        let lines = file_content.lines().collect::<Vec<&str>>();

        let syntax = syn::parse_file(&file_content).unwrap();

        for item in &syntax.items {
            let (mut f, mut s) = parse_item(
                item,
                TContext {
                    module: Some(
                        relative_path
                            .parent()
                            .unwrap()
                            .file_name()
                            .unwrap_or_default()
                            .to_str()
                            .unwrap()
                            .to_string(),
                    ),
                    file_path: Some(relative_path.to_str().unwrap().to_string()),
                    file_name: Some(path.file_name().unwrap().to_str().unwrap().to_string()),
                    struct_name: None,
                    snippet: None,
                },
                &lines,
            );
            functions.append(&mut f);
            structs.append(&mut s);
        }
    })
    .unwrap();

    structs.extend_from_slice(&functions);

    structs
}
