import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import {ApiAiClient} from 'api-ai-javascript'

 
@Injectable({
  providedIn: 'root'
})
export class ChatService {

  readonly token = environment.dialogflow.instituteBot
  readonly client = new ApiAiClient({accessToken:this.token})

  constructor() { }

  talk() {
    
  }
}
