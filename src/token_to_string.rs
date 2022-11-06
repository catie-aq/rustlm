use std::fmt;

#[derive(Clone, Copy, Debug)]
pub enum AlphabetError {
    UnknownAlphabet,
}

impl fmt::Display for AlphabetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlphabetError::UnknownAlphabet => {
                write!(f, "Unknown alphabet type")
            },
        }
    }
}

impl std::error::Error for AlphabetError {}

pub fn token_to_string(input: Vec<String>, alphabet_type: &String) -> Result<String, AlphabetError> {

    if "characters".to_string().eq(alphabet_type) {
        Ok(input.join("").clone())
    } else if "wordpiece".to_string().eq(alphabet_type) {
        let mut vec_str: String = "".to_string();
        for tok in input {
            let mut tok_chars = tok.chars();
            let mut first_char = tok_chars.nth(0).unwrap();
            if first_char != '[' {
                if first_char == '#' {
                    vec_str.push_str(&tok[2..]);
                } else {
                    vec_str.push_str(" ");
                    vec_str.push_str(&tok);
                }
            }
        }
        if vec_str.is_empty(){
            Ok(vec_str.to_string())
        }
        else{
            Ok(vec_str[1..].to_string()) // remove first character because of initial space
        }
    } else {
        return Err(AlphabetError::UnknownAlphabet);
    }

}
