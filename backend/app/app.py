from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from datetime import timedelta

from app.auth import create_access_token, get_current_user
from app.sentiment import analyze_text
from app.models import Token, TokenData, SentimentResponse

app = FastAPI(
    title="SentimentAI API",
    description="Sentiment analysis with JWT-based authentication",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "admin" and form_data.password == "password":
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.post("/analyze", response_model=SentimentResponse)
async def analyze(text: str, current_user: TokenData = Depends(get_current_user)):
    result = analyze_text(text)
    return result

@app.get("/")
def read_root():
    return {"message": "Welcome to SentimentAI"}
